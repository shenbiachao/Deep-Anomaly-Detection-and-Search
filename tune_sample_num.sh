for sample_num in 2;
do
  for check_num in 10 100;
  do
    python3 benchmark.py --check_num $check_num --sample_num $sample_num
  done
done