use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::full_model::{self, ModelForwardWorkspace, ModelWeights, TrainConfig};
use crate::layer::CompiledKernels;
use crate::model::ModelConfig;

#[derive(Debug, Clone, Copy)]
pub struct ForwardSchedulerConfig {
    pub max_streams: usize,
    pub queue_capacity: usize,
    pub use_lean_workspace: bool,
    pub memory_guard_frac: f32,
}

impl ForwardSchedulerConfig {
    pub fn benchmark(max_streams: usize, use_lean_workspace: bool) -> Self {
        Self {
            max_streams,
            queue_capacity: max_streams * 2,
            use_lean_workspace,
            memory_guard_frac: 0.85,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForwardBenchRequest {
    pub request_id: u64,
    pub tokens: Arc<[u32]>,
    pub targets: Arc<[u32]>,
}

#[derive(Debug, Clone)]
pub struct ForwardBenchResult {
    pub request_id: u64,
    pub loss: f32,
    pub queue_wait_ms: f32,
    pub service_ms: f32,
    pub total_ms: f32,
}

enum Job {
    Run {
        weights: Arc<ModelWeights>,
        request: ForwardBenchRequest,
        submitted_at: Instant,
        response: mpsc::Sender<ForwardBenchResult>,
    },
    Shutdown,
}

pub struct ForwardScheduler {
    tx: mpsc::SyncSender<Job>,
    handles: Vec<JoinHandle<()>>,
}

impl ForwardScheduler {
    pub fn new(cfg: ModelConfig, scheduler_cfg: ForwardSchedulerConfig) -> Self {
        let (tx, rx) = mpsc::sync_channel::<Job>(scheduler_cfg.queue_capacity);
        let rx = Arc::new(Mutex::new(rx));
        let tc = TrainConfig::default();

        let mut handles = Vec::with_capacity(scheduler_cfg.max_streams);
        for _ in 0..scheduler_cfg.max_streams {
            let rx = Arc::clone(&rx);
            let cfg = cfg.clone();
            handles.push(thread::spawn(move || {
                let kernels = if scheduler_cfg.use_lean_workspace {
                    CompiledKernels::compile_forward_only(&cfg)
                } else {
                    CompiledKernels::compile(&cfg)
                };
                let mut ws = if scheduler_cfg.use_lean_workspace {
                    ModelForwardWorkspace::new_lean(&cfg)
                } else {
                    ModelForwardWorkspace::new(&cfg)
                };

                loop {
                    let job = {
                        let guard = rx.lock().expect("multistream receiver poisoned");
                        guard.recv()
                    };
                    match job {
                        Ok(Job::Run {
                            weights,
                            request,
                            submitted_at,
                            response,
                        }) => {
                            let queue_wait_ms = submitted_at.elapsed().as_secs_f32() * 1000.0;
                            let t0 = Instant::now();
                            let loss = if scheduler_cfg.use_lean_workspace {
                                full_model::forward_only_ws(
                                    &cfg,
                                    &kernels,
                                    &weights,
                                    &request.tokens,
                                    &request.targets,
                                    tc.softcap,
                                    &mut ws,
                                )
                            } else {
                                full_model::forward_ws(
                                    &cfg,
                                    &kernels,
                                    &weights,
                                    &request.tokens,
                                    &request.targets,
                                    tc.softcap,
                                    &mut ws,
                                )
                            };
                            let service_ms = t0.elapsed().as_secs_f32() * 1000.0;
                            let total_ms = submitted_at.elapsed().as_secs_f32() * 1000.0;
                            let _ = response.send(ForwardBenchResult {
                                request_id: request.request_id,
                                loss,
                                queue_wait_ms,
                                service_ms,
                                total_ms,
                            });
                        }
                        Ok(Job::Shutdown) | Err(_) => break,
                    }
                }
            }));
        }

        Self { tx, handles }
    }

    pub fn run_batch(
        &self,
        weights: Arc<ModelWeights>,
        requests: Vec<ForwardBenchRequest>,
    ) -> Vec<ForwardBenchResult> {
        let (response_tx, response_rx) = mpsc::channel();
        let count = requests.len();
        for request in requests {
            self.tx
                .send(Job::Run {
                    weights: Arc::clone(&weights),
                    request,
                    submitted_at: Instant::now(),
                    response: response_tx.clone(),
                })
                .expect("enqueue forward bench request");
        }
        drop(response_tx);

        let mut results = Vec::with_capacity(count);
        for _ in 0..count {
            results.push(response_rx.recv().expect("receive forward bench result"));
        }
        results.sort_by_key(|r| r.request_id);
        results
    }
}

impl Drop for ForwardScheduler {
    fn drop(&mut self) {
        for _ in 0..self.handles.len() {
            let _ = self.tx.send(Job::Shutdown);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
