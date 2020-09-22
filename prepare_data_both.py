import os

# os.system('python scripts/extract_gemaps.py 0')
# os.system('python scripts/extract_gemaps.py 1')
# os.system('python scripts/prepare_gemaps.py 0') CAN USE PREPARED GEMAPS FROM INDIVIDUAL DATA SETS
# os.system('python scripts/prepare_gemaps.py 1')
# os.system('python scripts/get_word_timings.py') USE OUTPUT FILES FROM INDIVIDUAL DATA SETS INSTEAD
os.system('python scripts/prepare_fast_data_acous.py')
os.system('python scripts/get_vocab_both.py')
os.system('python scripts/get_word_annotations_both.py 0')
os.system('python scripts/get_word_annotations_both.py 1')
os.system('python scripts/get_averaged_annotations.py 0')
os.system('python scripts/get_averaged_annotations.py 1')
os.system('python scripts/prepare_fast_data_ling.py 0')
os.system('python scripts/prepare_fast_data_ling.py 1')
os.system('python scripts/find_pauses.py')
os.system('python scripts/find_overlaps.py')
os.system('python scripts/find_onsets.py')