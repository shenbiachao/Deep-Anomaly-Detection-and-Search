for reward in 0.1;
do
  for check_num in 15 100;
  do
    python3 benchmark.py --check_num $check_num --reward_list 1.0 -1.0 1.0 $reward
  done
done
