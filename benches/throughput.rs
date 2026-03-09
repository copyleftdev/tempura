use criterion::{criterion_group, criterion_main, Criterion};

fn throughput_benchmark(_c: &mut Criterion) {
    // TODO: Implement proposals/second benchmark
}

criterion_group!(benches, throughput_benchmark);
criterion_main!(benches);
