
# generate SIG
# python3 -u src/backdoor_imagenet_generation_for_eval.py --name=backdoor_imagenet_generation_for_eval --device_id=3 --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type SIG --save_files_name=ILSVRC2012_SIG_val

# test SIG asr
# --add_backdoor --asr --label banana --patch_type SIG --save_files_name=ILSVRC2012_SIG_val

# test SSBA asr
# --add_backdoor --asr --label banana --patch_type issba --save_files_name=ILSVRC2012_issba_val