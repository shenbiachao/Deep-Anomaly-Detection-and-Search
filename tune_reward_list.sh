for reward in 0.8;
do
  for check_num in 100 10;
  do
    python3 benchmark.py --check_num $check_num --reward_list 1.0 -1.0 1.0 $reward
  done
done