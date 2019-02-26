python convert_to_rouge.py --ref_data_dir rouge/ref_data_path/ --gen_data_dir rouge/gen_data_path/ --test_number 100

ROUGE-1.5.5/ROUGE-1.5.5.pl -e ROUGE-1.5.5/data/ -n 4 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 config.xml > result.txt None