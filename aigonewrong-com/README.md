```
# zappa with venv

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.8-dev python3.8-venv -yq

python3.8 -m venv myvenv
source myvenv/bin/activate

pip install -r requirements.txt



zappa init

-- stage: prod
-- lambda name: zappa-aigonewrong-com

zappa deploy prod
zappa update prod


https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/Welcome.html

https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-to-api-gateway.html

# aws stuff


register domain via aws route53: https://us-east-1.console.aws.amazon.com/route53

create cert via aws acm: https://us-east-1.console.aws.amazon.com/acm
had to click "create record in route53

once lambda deploy via zappa, goto coresponding apigateway: https://us-east-1.console.aws.amazon.com/apigateway
+ click `Custom domain names`
+ create domainname `www.aigonewrong.com` and associate it with the just created ACM certificate ( cert may take 10 min to get created/approved )
+ once created, use `Configure API mappings` to map to lambda

+ go back to route53, and goto hostedzones->CreateRecord, follow below to create route to api gateway ( use simple record )
```
www.aigonewrong.com
Record type: A 
Route to: api gateway us-east-1 xxxxxxxx.execute-api...
```
https://hackernoon.com/how-to-setup-subdomain-for-aws-api-gateway-d526a9fd6722
https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-to-api-gateway.html





```