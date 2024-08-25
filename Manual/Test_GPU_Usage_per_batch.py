## Test GPU Usage per batch

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated() / 1024**2  # Convert bytes to megabytes

# Function to measure the memory usage of a batch
def measure_batch_memory_usage(data_loader, device, num_batches=5):
    initial_memory = get_gpu_memory_usage()
    print(f"Initial GPU memory usage: {initial_memory:.2f} MB")

    batch_memory_usages = []
    for i, (spectrograms, labels) in enumerate(data_loader):
        if i >= num_batches:
            break

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Measure memory usage before loading the batch
        before_batch_memory = get_gpu_memory_usage()

        spectrograms, labels = spectrograms.to(device), labels.to(device)

        # Measure memory usage after loading the batch
        after_batch_memory = get_gpu_memory_usage()

        batch_memory_usage = after_batch_memory - before_batch_memory
        batch_memory_usages.append(batch_memory_usage)
        print(f"Batch {i+1} memory usage: {batch_memory_usage:.2f} MB")

        # Clear GPU memory after measuring
        torch.cuda.empty_cache()

    return batch_memory_usages

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Measure memory usage for the first 5 batches
batch_memory_usages = measure_batch_memory_usage(train_loader, device, num_batches=5)
print(f"Average memory usage per batch: {sum(batch_memory_usages)/len(batch_memory_usages):.2f} MB")