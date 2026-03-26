use std::process::{Child, Command, ExitStatus};
use std::thread;
use std::time::{Duration, Instant};

fn run_child(args: &[&str]) -> (std::process::ExitStatus, Duration) {
    let start = Instant::now();
    let status = Command::new(env!("CARGO_BIN_EXE_theory_runner"))
        .args(args)
        .status()
        .expect("spawn theory_runner");
    (status, start.elapsed())
}

fn run_child_with_timeout(args: &[&str], timeout: Duration) -> (ExitStatus, Duration) {
    let start = Instant::now();
    let mut child: Child = Command::new(env!("CARGO_BIN_EXE_theory_runner"))
        .args(args)
        .spawn()
        .expect("spawn theory_runner");

    loop {
        if let Some(status) = child.try_wait().expect("poll theory_runner") {
            return (status, start.elapsed());
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let status = child.wait().expect("wait theory_runner after kill");
            return (status, start.elapsed());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn theory_runner_noop_exits_zero() {
    let (status, elapsed) = run_child(&["noop"]);
    assert!(status.success(), "noop should succeed");
    assert!(
        elapsed < Duration::from_secs(5),
        "noop should return quickly"
    );
}

#[test]
fn theory_runner_thread_panic_aborts_process() {
    let (status, elapsed) = run_child(&["panic-thread"]);
    assert!(!status.success(), "panic-thread should fail");
    assert!(
        elapsed < Duration::from_secs(5),
        "panic-thread should terminate quickly"
    );
}

#[test]
#[ignore]
fn theory_forward_7b_isolated() {
    let (status, elapsed) = run_child_with_timeout(&["forward-7b"], Duration::from_secs(180));
    assert!(status.success(), "isolated 7B forward should succeed");
    assert!(
        elapsed < Duration::from_secs(180),
        "isolated 7B forward should complete within timeout"
    );
}

#[test]
#[ignore]
fn theory_forward_10b_isolated() {
    let (status, elapsed) = run_child_with_timeout(&["forward-10b"], Duration::from_secs(240));
    assert!(status.success(), "isolated 10B forward should succeed");
    assert!(
        elapsed < Duration::from_secs(240),
        "isolated 10B forward should complete within timeout"
    );
}
