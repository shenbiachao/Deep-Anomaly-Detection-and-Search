for sampling_distribution in 1.0 0.0;
do
  for check_num in 10 100;
  do
    python3 benchmark.py --check_num $check_num --sampling_method_distribution $sampling_distribution
  done
done