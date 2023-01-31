for search_percentage in 1.0 0.5;
do
  for check_num in 15 100;
  do
    python3 benchmark.py --check_num $check_num --search_percentage $search_percentage
  done
done
