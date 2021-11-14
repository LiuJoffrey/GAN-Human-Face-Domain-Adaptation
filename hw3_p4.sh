# TODO: create shell script for running your improved UDA model

cd p4

if [ "$2" = "usps" ]
then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vEtvDNDGTJM7KORTAiNJO_W0l4q2Ztnp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vEtvDNDGTJM7KORTAiNJO_W0l4q2Ztnp" -O svhn_usps.zip && rm -rf /tmp/cookies.txt
    unzip svhn_usps.zip
fi

if [ "$2" = "mnistm" ]
then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=118rh4N8neUavxSi0VxU97IZvnAfUOeii' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=118rh4N8neUavxSi0VxU97IZvnAfUOeii" -O usps_mnistm_1.zip && rm -rf /tmp/cookies.txt
    unzip usps_mnistm_1.zip
fi

if [ "$2" = "svhn" ]
then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nNYj56kMQKWv75f-0kLd5Br2sRtKFJCJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nNYj56kMQKWv75f-0kLd5Br2sRtKFJCJ" -O mnistm_svhn_1.zip && rm -rf /tmp/cookies.txt
    unzip mnistm_svhn_1.zip
fi

python3 test.py $1 $2 $3

