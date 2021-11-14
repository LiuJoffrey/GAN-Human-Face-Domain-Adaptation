# TODO: create shell script for running your GAN/ACGAN model

cd p1p2


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12hD56J2VHh-R4ab7f5ftFBZBJHhk8f-q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12hD56J2VHh-R4ab7f5ftFBZBJHhk8f-q" -O G_ACG2_model.pkt.196 && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NYqLpUQ2WyYSqQ4MuhjHF38LnpXpCNod' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NYqLpUQ2WyYSqQ4MuhjHF38LnpXpCNod" -O G_model.pkt.105 && rm -rf /tmp/cookies.txt

python3 GAN_inference.py $1
python3 ACGAN_inference.py $1

