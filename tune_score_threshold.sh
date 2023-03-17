for score_threshold in 0.01 0.99;
do
  for check_num in 10 100;
  do
    python3 benchmark.py --check_num $check_num --score_threshold $score_threshold
  done
done
