for sampling in 0.5;
do
  for check_num in 15 100;
  do
    python3 benchmark.py --check_num $check_num --sampling_method_distribution $sampling
  done
done
