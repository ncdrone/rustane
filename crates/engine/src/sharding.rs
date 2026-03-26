use crate::model::ModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardPolicy {
    FailFast,
    AutoAdjustNearest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShardRequest {
    pub attn_fwd_shards: Option<usize>,
    pub attn_bwd_shards: Option<usize>,
    pub ffn_fwd_shards: Option<usize>,
    pub ffn_bwd_shards: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolvedForwardMode {
    #[default]
    Baseline,
    AttentionOnly {
        attn_shards: usize,
    },
    FfnOnly {
        ffn_shards: usize,
    },
    AttentionFfn {
        attn_shards: usize,
        ffn_shards: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolvedBackwardMode {
    #[default]
    Baseline,
    AttentionOnly {
        attn_shards: usize,
    },
    FfnOnly {
        ffn_shards: usize,
    },
    AttentionFfn {
        attn_shards: usize,
        ffn_shards: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModeAdjustment {
    pub axis: &'static str,
    pub requested: usize,
    pub applied: usize,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModeResolution {
    pub forward: ResolvedForwardMode,
    pub backward: ResolvedBackwardMode,
    pub adjustments: Vec<ModeAdjustment>,
}

impl ShardRequest {
    pub fn from_env() -> Result<Self, String> {
        let attn_shorthand = parse_env_count("ATTN_SHARDS")?;
        let ffn_shorthand = parse_env_count("FFN_SHARDS")?;
        let attn_fwd = parse_env_count("ATTN_FWD_SHARDS")?;
        let attn_bwd = parse_env_count("ATTN_BWD_SHARDS")?;
        let ffn_fwd = parse_env_count("FFN_FWD_SHARDS")?;
        let ffn_bwd = parse_env_count("FFN_BWD_SHARDS")?;

        if let Some(value) = attn_shorthand {
            if attn_fwd.is_some() || attn_bwd.is_some() {
                return Err(format!(
                    "ATTN_SHARDS={value} conflicts with per-direction attention shard vars; use per-direction vars or the shorthand, not both"
                ));
            }
        }
        if let Some(value) = ffn_shorthand {
            if ffn_fwd.is_some() || ffn_bwd.is_some() {
                return Err(format!(
                    "FFN_SHARDS={value} conflicts with per-direction FFN shard vars; use per-direction vars or the shorthand, not both"
                ));
            }
        }

        Ok(Self {
            attn_fwd_shards: attn_fwd.or(attn_shorthand),
            attn_bwd_shards: attn_bwd.or(attn_shorthand),
            ffn_fwd_shards: ffn_fwd.or(ffn_shorthand),
            ffn_bwd_shards: ffn_bwd.or(ffn_shorthand),
        })
    }

    pub fn from_forward_requests(attn_request: Option<usize>, ffn_request: Option<usize>) -> Self {
        Self {
            attn_fwd_shards: attn_request,
            attn_bwd_shards: None,
            ffn_fwd_shards: ffn_request,
            ffn_bwd_shards: None,
        }
    }
}

impl ResolvedForwardMode {
    pub fn is_baseline(self) -> bool {
        matches!(self, Self::Baseline)
    }

    pub fn attn_shards(self) -> usize {
        match self {
            Self::Baseline | Self::FfnOnly { .. } => 1,
            Self::AttentionOnly { attn_shards } | Self::AttentionFfn { attn_shards, .. } => {
                attn_shards
            }
        }
    }

    pub fn ffn_shards(self) -> usize {
        match self {
            Self::Baseline | Self::AttentionOnly { .. } => 1,
            Self::FfnOnly { ffn_shards } | Self::AttentionFfn { ffn_shards, .. } => ffn_shards,
        }
    }

    pub fn mode_label(self) -> String {
        match self {
            Self::Baseline => "baseline".to_string(),
            Self::AttentionOnly { attn_shards } => format!("attn{attn_shards}"),
            Self::FfnOnly { ffn_shards } => format!("ffn{ffn_shards}"),
            Self::AttentionFfn {
                attn_shards,
                ffn_shards,
            } => format!("attn{attn_shards}_ffn{ffn_shards}"),
        }
    }
}

impl ResolvedBackwardMode {
    pub fn is_baseline(self) -> bool {
        matches!(self, Self::Baseline)
    }

    pub fn attn_shards(self) -> usize {
        match self {
            Self::Baseline | Self::FfnOnly { .. } => 1,
            Self::AttentionOnly { attn_shards } | Self::AttentionFfn { attn_shards, .. } => {
                attn_shards
            }
        }
    }

    pub fn ffn_shards(self) -> usize {
        match self {
            Self::Baseline | Self::AttentionOnly { .. } => 1,
            Self::FfnOnly { ffn_shards } | Self::AttentionFfn { ffn_shards, .. } => ffn_shards,
        }
    }
}

pub fn resolve_modes(
    cfg: &ModelConfig,
    requested: ShardRequest,
    policy: ShardPolicy,
) -> Result<ModeResolution, String> {
    let mut adjustments = Vec::new();
    let attn_valid = divisors(gcd(cfg.heads, cfg.kv_heads));
    let ffn_valid = divisors(cfg.hidden);

    let attn_fwd = resolve_axis(
        "ATTN_FWD_SHARDS",
        "attention group count",
        gcd(cfg.heads, cfg.kv_heads),
        requested.attn_fwd_shards,
        &attn_valid,
        policy,
        &mut adjustments,
    )?;
    let ffn_fwd = resolve_axis(
        "FFN_FWD_SHARDS",
        "hidden width",
        cfg.hidden,
        requested.ffn_fwd_shards,
        &ffn_valid,
        policy,
        &mut adjustments,
    )?;
    let attn_bwd = resolve_axis(
        "ATTN_BWD_SHARDS",
        "attention group count",
        gcd(cfg.heads, cfg.kv_heads),
        requested.attn_bwd_shards,
        &attn_valid,
        policy,
        &mut adjustments,
    )?;
    let ffn_bwd = resolve_axis(
        "FFN_BWD_SHARDS",
        "hidden width",
        cfg.hidden,
        requested.ffn_bwd_shards,
        &ffn_valid,
        policy,
        &mut adjustments,
    )?;

    Ok(ModeResolution {
        forward: build_forward_mode(attn_fwd, ffn_fwd),
        backward: build_backward_mode(attn_bwd, ffn_bwd),
        adjustments,
    })
}

fn parse_env_count(var: &'static str) -> Result<Option<usize>, String> {
    let Some(raw) = std::env::var(var).ok() else {
        return Ok(None);
    };
    let value = raw
        .parse::<usize>()
        .map_err(|_| format!("{var} must be an integer, got '{raw}'"))?;
    if value == 0 {
        return Err(format!("{var} must be positive"));
    }
    Ok(Some(value))
}

fn resolve_axis(
    axis: &'static str,
    shape_name: &str,
    shape_value: usize,
    requested: Option<usize>,
    valid_divisors: &[usize],
    policy: ShardPolicy,
    adjustments: &mut Vec<ModeAdjustment>,
) -> Result<usize, String> {
    let Some(requested) = requested else {
        return Ok(1);
    };
    if valid_divisors.contains(&requested) {
        return Ok(requested);
    }

    match policy {
        ShardPolicy::FailFast => Err(format!(
            "{axis}={requested} is invalid for {shape_name}={shape_value}; valid divisors: {}",
            join_usizes(valid_divisors)
        )),
        ShardPolicy::AutoAdjustNearest => {
            let applied = nearest_divisor(valid_divisors, requested);
            adjustments.push(ModeAdjustment {
                axis,
                requested,
                applied,
                reason: format!("nearest valid divisor for {shape_name}={shape_value}"),
            });
            Ok(applied)
        }
    }
}

fn build_forward_mode(attn_shards: usize, ffn_shards: usize) -> ResolvedForwardMode {
    match (attn_shards > 1, ffn_shards > 1) {
        (false, false) => ResolvedForwardMode::Baseline,
        (true, false) => ResolvedForwardMode::AttentionOnly { attn_shards },
        (false, true) => ResolvedForwardMode::FfnOnly { ffn_shards },
        (true, true) => ResolvedForwardMode::AttentionFfn {
            attn_shards,
            ffn_shards,
        },
    }
}

fn build_backward_mode(attn_shards: usize, ffn_shards: usize) -> ResolvedBackwardMode {
    match (attn_shards > 1, ffn_shards > 1) {
        (false, false) => ResolvedBackwardMode::Baseline,
        (true, false) => ResolvedBackwardMode::AttentionOnly { attn_shards },
        (false, true) => ResolvedBackwardMode::FfnOnly { ffn_shards },
        (true, true) => ResolvedBackwardMode::AttentionFfn {
            attn_shards,
            ffn_shards,
        },
    }
}

fn divisors(value: usize) -> Vec<usize> {
    let mut divisors = Vec::new();
    let mut i = 1usize;
    while i * i <= value {
        if value % i == 0 {
            divisors.push(i);
            if i != value / i {
                divisors.push(value / i);
            }
        }
        i += 1;
    }
    divisors.sort_unstable();
    divisors
}

fn nearest_divisor(divisors: &[usize], requested: usize) -> usize {
    *divisors
        .iter()
        .min_by_key(|&&divisor| (requested.abs_diff(divisor), divisor))
        .expect("at least one divisor")
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

fn join_usizes(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(heads: usize, kv_heads: usize, hidden: usize) -> ModelConfig {
        ModelConfig {
            dim: 1024,
            hidden,
            heads,
            kv_heads,
            hd: 128,
            seq: 512,
            nlayers: 8,
            vocab: 8192,
            q_dim: heads * 128,
            kv_dim: kv_heads * 128,
            gqa_ratio: heads / kv_heads,
        }
    }

    #[test]
    fn resolve_modes_fail_fast_reports_divisors() {
        let err = resolve_modes(
            &cfg(40, 40, 11008),
            ShardRequest::from_forward_requests(Some(3), Some(10)),
            ShardPolicy::FailFast,
        )
        .unwrap_err();
        assert!(err.contains("valid divisors"));
    }

    #[test]
    fn resolve_modes_auto_adjust_records_adjustments() {
        let resolution = resolve_modes(
            &cfg(40, 40, 11008),
            ShardRequest::from_forward_requests(Some(9), Some(10)),
            ShardPolicy::AutoAdjustNearest,
        )
        .expect("resolve modes");
        assert_eq!(resolution.forward.attn_shards(), 8);
        assert_eq!(resolution.forward.ffn_shards(), 8);
        assert_eq!(resolution.adjustments.len(), 2);
    }
}
