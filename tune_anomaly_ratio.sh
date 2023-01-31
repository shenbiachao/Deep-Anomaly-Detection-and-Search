for anomaly_ratio in 0.4 0.5 1.0;
do
  for check_num in 10 100;
  do
    python3 benchmark.py --check_num $check_num --anomaly_ratio $anomaly_ratio
  done
done
