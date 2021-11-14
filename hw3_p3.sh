# TODO: create shell script for running your DANN model

cd p3

if [ "$2" = "usps" ]
then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12SbgNzMTOVY423a1LCAc8d2BOMS87vTn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12SbgNzMTOVY423a1LCAc8d2BOMS87vTn" -O svhn_usps_model.pth && rm -rf /tmp/cookies.txt
fi

if [ "$2" = "mnistm" ]
then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1J8avQdkFV3OgWuMZH-FIzM9TI83KXofT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1J8avQdkFV3OgWuMZH-FIzM9TI83KXofT" -O usps_mnistm_model_42.pth && rm -rf /tmp/cookies.txt
fi

if [ "$2" = "svhn" ]
then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qxgSGsb-kMW0ov8fT7knSDURuPaM2zw6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qxgSGsb-kMW0ov8fT7knSDURuPaM2zw6" -O mnistm_svhn_model.pth && rm -rf /tmp/cookies.txt
fi

python3 test.py $1 $2 $3


