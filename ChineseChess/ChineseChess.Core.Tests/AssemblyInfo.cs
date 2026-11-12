using Xunit;

// Allocation-based assertions read GC.GetTotalAllocatedBytes, which is a
// process-wide counter. Parallel test collections would pollute the measurement.
[assembly: CollectionBehavior(DisableTestParallelization = true)]