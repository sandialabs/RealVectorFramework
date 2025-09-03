# Benchmarks on Samsung Galaxy Book Pro

    --------------------------------------------------------------------------------------------------
    Benchmark                                        Time             CPU   Iterations UserCounters...
    --------------------------------------------------------------------------------------------------
    BM_RegularClone_Small                         35.4 ns         35.4 ns     19675872 items_per_second=28.2461M/s 100 elements
    BM_ArenaClone_Small                           65.1 ns         65.1 ns     10527215 items_per_second=15.3661M/s 100 elements
    BM_RegularClone_Medium                        1993 ns         1992 ns       351396 items_per_second=501.914k/s 10K elements
    BM_ArenaClone_Medium                          65.7 ns         65.7 ns     10637911 items_per_second=15.2235M/s 10K elements
    BM_RegularClone_Large                       358504 ns       358372 ns         1954 items_per_second=2.7904k/s 1M elements
    BM_ArenaClone_Large                           65.0 ns         64.9 ns     10727916 items_per_second=15.3972M/s 1M elements
    BM_RegularClone_Batch/1000/10                 1885 ns         1885 ns       370218 items_per_second=5.30543M/s 1000 elem, batch 10
    BM_RegularClone_Batch/1000/100               19185 ns        19178 ns        36548 items_per_second=5.21427M/s 1000 elem, batch 100
    BM_RegularClone_Batch/1000/1000             335281 ns       334994 ns         2079 items_per_second=2.98513M/s 1000 elem, batch 1000
    BM_RegularClone_Batch/10000/10               20190 ns        20183 ns        34559 items_per_second=495.472k/s 10000 elem, batch 10
    BM_RegularClone_Batch/10000/100             347120 ns       347005 ns         2008 items_per_second=288.181k/s 10000 elem, batch 100
    BM_RegularClone_Batch/100000/10             366671 ns       366549 ns         1910 items_per_second=27.2815k/s 100000 elem, batch 10
    BM_ArenaClone_Batch/1000/10                    682 ns          682 ns      1026636 items_per_second=14.659M/s 1000 elem, batch 10
    BM_ArenaClone_Batch/1000/100                  6493 ns         6489 ns       107283 items_per_second=15.411M/s 1000 elem, batch 100
    BM_ArenaClone_Batch/1000/1000                64867 ns        64809 ns        10769 items_per_second=15.43M/s 1000 elem, batch 1000
    BM_ArenaClone_Batch/10000/10                   692 ns          691 ns      1012675 items_per_second=14.4805M/s 10000 elem, batch 10
    BM_ArenaClone_Batch/10000/100                 6554 ns         6552 ns       107295 items_per_second=15.2635M/s 10000 elem, batch 100
    BM_ArenaClone_Batch/100000/10                  690 ns          690 ns      1017245 items_per_second=14.5015M/s 100000 elem, batch 10
    BM_RegularClone_AlgorithmPattern/1000         5673 ns         5663 ns       130888 items_per_second=5.29799M/s 1000 elements
    BM_RegularClone_AlgorithmPattern/10000       59931 ns        59884 ns        11623 items_per_second=500.97k/s 10000 elements
    BM_RegularClone_AlgorithmPattern/100000    1090952 ns      1089213 ns          644 items_per_second=27.5428k/s 100000 elements
    BM_ArenaClone_AlgorithmPattern/1000           1909 ns         1909 ns       367054 items_per_second=15.7183M/s 1000 elements
    BM_ArenaClone_AlgorithmPattern/10000          1912 ns         1911 ns       365698 items_per_second=15.6959M/s 10000 elements
    BM_ArenaClone_AlgorithmPattern/100000         1905 ns         1904 ns       367395 items_per_second=15.7557M/s 100000 elements
    BM_Arena_Statistics                           5.85 ns         5.85 ns    119932589
    BM_Arena_PoolStats                            54.6 ns         54.6 ns     12817296
    BM_ArenaWithObservers_Clone                    134 ns          134 ns      5222818 items_per_second=7.45651M/s
    BM_Arena_MemoryPressure/10                     689 ns          689 ns      1015910 items_per_second=14.5087M/s
    BM_Arena_MemoryPressure/100                   6589 ns         6587 ns       106184 items_per_second=15.1817M/s
    BM_Arena_MemoryPressure/1000                 66123 ns        66100 ns        10580 items_per_second=15.1286M/s
    BM_Arena_MemoryPressure/10000               742703 ns       742449 ns          942 items_per_second=13.4689M/s


    ----------------------------------------------------------------------------------------------------------
    Benchmark                                                Time             CPU   Iterations UserCounters...
    ----------------------------------------------------------------------------------------------------------
    BM_StackVector_Clone<100>                             17.5 ns         17.5 ns     38038735 items_per_second=57.0191M/s 100 elements (stack)
    BM_HeapVector_Clone/100                               23.2 ns         23.2 ns     30214517 items_per_second=43.1161M/s 100 elements (heap)
    BM_HeapVector_Arena/100                               61.8 ns         61.8 ns     11362948 items_per_second=16.1916M/s 100 elements (arena)
    BM_StackVector_Workspace<100>                         1.30 ns         1.30 ns    538927819 items_per_second=770.743M/s 100 elements (workspace)
    BM_StackVector_Clone<1000>                            66.4 ns         66.4 ns     10722350 items_per_second=15.0655M/s 1000 elements (stack)
    BM_HeapVector_Clone/1000                              86.3 ns         86.3 ns      8086253 items_per_second=11.5913M/s 1000 elements (heap)
    BM_HeapVector_Arena/1000                              61.6 ns         61.6 ns     11373307 items_per_second=16.2342M/s 1000 elements (arena)
    BM_StackVector_Workspace<1000>                        1.30 ns         1.30 ns    539922515 items_per_second=771.229M/s 1000 elements (workspace)
    BM_HeapVector_Clone/10000                             1997 ns         1997 ns       350429 items_per_second=500.756k/s 10000 elements (heap)
    BM_HeapVector_Arena/10000                             61.6 ns         61.6 ns     11394889 items_per_second=16.2347M/s 10000 elements (arena)
    BM_HeapVector_Clone/100000                           31757 ns        31750 ns        22088 items_per_second=31.4957k/s 100000 elements (heap)
    BM_HeapVector_Arena/100000                            61.6 ns         61.6 ns     11351381 items_per_second=16.2274M/s 100000 elements (arena)
    BM_StackVector_AlgorithmPattern<100>                   518 ns          518 ns      1353342 items_per_second=57.9586M/s 100 elements (stack)
    BM_HeapVector_AlgorithmPattern/100                    1164 ns         1164 ns       602381 items_per_second=25.7694M/s 100 elements (heap)
    BM_HeapVector_Arena_AlgorithmPattern/100              1850 ns         1850 ns       378738 items_per_second=16.2166M/s 100 elements (arena)
    BM_StackVector_Workspace_AlgorithmPattern<100>        62.0 ns         62.0 ns     11197947 items_per_second=483.936M/s 100 elements (workspace)
    BM_StackVector_AlgorithmPattern<1000>                 2055 ns         2054 ns       336946 items_per_second=14.6027M/s 1000 elements (stack)
    BM_HeapVector_AlgorithmPattern/1000                   3643 ns         3643 ns       192125 items_per_second=8.23565M/s 1000 elements (heap)
    BM_HeapVector_Arena_AlgorithmPattern/1000             1853 ns         1852 ns       377377 items_per_second=16.1954M/s 1000 elements (arena)
    BM_StackVector_Workspace_AlgorithmPattern<1000>       61.6 ns         61.5 ns     11299681 items_per_second=487.427M/s 1000 elements (workspace)
    BM_HeapVector_AlgorithmPattern/10000                 60691 ns        60679 ns        11534 items_per_second=494.408k/s 10000 elements (heap)
    BM_HeapVector_Arena_AlgorithmPattern/10000            1858 ns         1857 ns       377946 items_per_second=16.1538M/s 10000 elements (arena)
    BM_HeapVector_AlgorithmPattern/100000              5917110 ns      5915888 ns          115 items_per_second=5.07109k/s 100000 elements (heap)
    BM_HeapVector_Arena_AlgorithmPattern/100000           1860 ns         1859 ns       377759 items_per_second=16.1337M/s 100000 elements (arena)
