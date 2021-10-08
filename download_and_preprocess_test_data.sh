mkdir data/ru-de
mkdir data/de-zh

for lp in "de-zh" "et-en" "ro-en" "ru-de"
do
	wget -c "https://github.com/eval4nlp/SharedTask2021/raw/main/data/test21/${lp}-test21.tar.gz"
	tar -zxvf "${lp}-test21.tar.gz"
	rm ${lp}-test21.tar.gz
	mv ${lp}-test21/* data/${lp}/
	rm -rf ${lp}-test21/
done
