import torch
import time
import yaml
import pandas as pd
from torchvision import models

# ------------------------------
# CONFIG
# ------------------------------
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
PRECISIONS = ['fp32', 'fp16']  # FP16 only on GPU

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
CONFIG_PATH = '../config/models.yaml'
OUTPUT_CSV = '../results/data/benchmark_results.csv'

NUM_WARMUP = 10
NUM_ITER = 50

# ------------------------------
# HELPER FUNCTION
# ------------------------------
def measure_latency(model, device, input_tensor, precision='fp32', num_iterations=NUM_ITER):
    """
    Measures forward pass latency for a model with a given input tensor.
    Handles FP16 safely with automatic mixed precision (AMP) on GPU.
    Returns a list of latencies in milliseconds.
    """
    latencies = []
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            if precision == 'fp16' and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark iterations
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()

            # Forward pass
            if precision == 'fp16' and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)

            # Measure elapsed
            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)  # ms
            else:
                elapsed = (time.perf_counter() - start_time) * 1000  # ms

            latencies.append(elapsed)

    return latencies

# ------------------------------
# MAIN BENCHMARK
# ------------------------------
def main():
    # Load models from YAML
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    results = []

    for model_name in config['models']:
        print(f"\n=== Benchmarking {model_name} ===")
        # Load pretrained model
        model = getattr(models, model_name)(weights=None).to(DEVICE)

        for batch_size in BATCH_SIZES:
            # Create dummy input
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=DEVICE, dtype=torch.float32)

            for precision in PRECISIONS:
                if precision == 'fp16' and DEVICE.type != 'cuda':
                    continue  # Skip FP16 on CPU

                # Measure latencies
                lat_list = measure_latency(model, DEVICE, input_tensor, precision)
                lat_tensor = torch.tensor(lat_list)

                # Compute throughput (images/sec)
                throughput = (batch_size * 1000) / lat_tensor.mean().item()

                # Compute P95, P99 safely
                p95 = lat_tensor.kthvalue(int(0.95 * len(lat_tensor))).values.item()
                p99 = lat_tensor.kthvalue(int(0.99 * len(lat_tensor))).values.item()

                # Append result
                results.append({
                    'model': model_name,
                    'device': str(DEVICE),
                    'precision': precision,
                    'batch_size': batch_size,
                    'mean_latency_ms': lat_tensor.mean().item(),
                    'median_latency_ms': lat_tensor.median().item(),
                    'p95_latency_ms': p95,
                    'p99_latency_ms': p99,
                    'throughput_img_per_sec': throughput
                })

                print(f"Batch {batch_size} | {precision} | Mean: {lat_tensor.mean():.2f} ms | Throughput: {throughput:.2f} img/s")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Benchmark completed! Results saved to {OUTPUT_CSV}")

# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == '__main__':
    main()
