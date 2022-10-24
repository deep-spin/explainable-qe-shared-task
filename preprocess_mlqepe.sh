for lp in "en-de" "en-zh" "et-en" "ne-en" "ro-en" "ru-en" "si-en"
do
	tar -zxvf mlqe-pe/data/direct-assessments/train/${lp}-train.tar.gz -C mlqe-pe/data/direct-assessments/train/
	tar -zxvf mlqe-pe/data/direct-assessments/dev/${lp}-dev.tar.gz -C mlqe-pe/data/direct-assessments/dev/
	tar -zxvf mlqe-pe/data/direct-assessments/test/${lp}-test.tar.gz -C mlqe-pe/data/direct-assessments/test/
done


for lp in "en-de" "en-zh" "et-en" "ne-en" "ro-en" "ru-en" "si-en"
do
	tar -zxvf mlqe-pe/data/post-editing/train/${lp}-train.tar.gz -C mlqe-pe/data/post-editing/train/
	tar -zxvf mlqe-pe/data/post-editing/dev/${lp}-dev.tar.gz -C mlqe-pe/data/post-editing/dev/
	tar -zxvf mlqe-pe/data/post-editing/test/${lp}-test.tar.gz -C mlqe-pe/data/post-editing/test/
done


# cd mlqe-pe/data/direct-assessments/train
# cat *.tar.gz | tar -zxvf - -i
# rm *.tar.gz
# cd ../test
# cat *.tar.gz | tar -zxvf - -i
# rm *.tar.gz
# cd ../dev
# cat *.tar.gz | tar -zxvf - -i
# rm *.tar.gz
# cd ../../../..
# cd mlqe-pe/data/post-editing/train
# cat *.tar.gz | tar -zxvf - -i
# rm *.tar.gz
# cd ../test
# cat *.tar.gz | tar -zxvf - -i
# rm *.tar.gz
# cd ../dev
# cat *.tar.gz | tar -zxvf - -i
# rm *.tar.gz
# cd ../../../..
