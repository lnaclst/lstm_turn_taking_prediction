import os

# this assumes same file structure as the original Roddy code, but with the following changes:
    # ./data/splits/complete.txt should be list of switchboard conversations
    # ./data/signals should contain switchboard wavs (remember to split channels first using
        # split_channels_switchboard.sh and then put the resulting wav files into ./data/signals/dialogues_mono
    # need to change annotations directory to point to the location of your switchboard nxt terminals folder in some of the files
        # /switchboard/nxt/xml/terminals/'

# os.system('python scripts/extract_gemaps.py 0')
# os.system('python scripts/extract_gemaps.py 1')
os.system('python scripts/prepare_gemaps.py 0')
os.system('python scripts/prepare_gemaps.py 1')
os.system('python scripts/prepare_fast_data_acous.py')
os.system('python scripts/get_word_timings.py')
os.system('python scripts/get_vocab_switchboard.py')
os.system('python scripts/get_word_annotations_switchboard.py 0')
os.system('python scripts/get_word_annotations_switchboard.py 1')
os.system('python scripts/get_averaged_annotations.py 0')
os.system('python scripts/get_averaged_annotations.py 1')
os.system('python scripts/prepare_fast_data_ling.py 0')
os.system('python scripts/prepare_fast_data_ling.py 1')
os.system('python scripts/find_pauses.py')
os.system('python scripts/find_overlaps.py')
os.system('python scripts/find_onsets.py')
