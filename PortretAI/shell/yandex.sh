curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash

exec -l $SHELL

curl -d "{\"yandexPassportOauthToken\":\"y0_AgAAAABceoMYAATuwQAAAAD7ckTLAADA4u8ywLdBh5fCISoRvUwV6kSpCQ\"}" "https://iam.api.cloud.yandex.net/iam/v1/tokens"

yc iam key create --service-account-name yg --output key.json

yc iam key create --folder-name aje0qr9uht19tsp96rdf --service-account-name yg --output key.json

yc iam key create --service-account-id aje0qr9uht19tsp96rdf --output key.json

yc config profile create yandexgpt-profile

yc config set service-account-key key.json



