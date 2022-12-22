```
# zappa with venv

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.8-dev python3.8-venv -yq

python3.8 -m venv ~/virtual_env/agw-com
source ~/virtual_env/agw-com/bin/activate

pip install -r requirements.txt



zappa init

-- stage: prod
-- lambda name: zappa-aigonewrong-com

echo $(openssl rand -hex 4)
append above to s3_bucket in `zappa_settings.json`

zappa deploy prod
zappa update prod


https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/Welcome.html

https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-to-api-gateway.html

# aws stuff


+register domain via aws route53: https://us-east-1.console.aws.amazon.com/route53
create cert via aws acm: https://us-east-1.console.aws.amazon.com/acm
had to click "create record in route53

+ create a hosted zone for above domain
aws will add 2 records, Type "NS" and "SOA"
verify via dig command:
```
dig SOA aigonwrong.com
dig NS aigonwrong.com
```
+ if response have 0 answers, ensure "Value/Route traffic to" for type NS
is copied over to "Name servers " in "Route 53->Registered domains->"aigonewrong.com".

then you should get ANSWERS for above 2.

+ create a wild card certificate
*.aigonewrong.com
+ then click "Create records in Route 53 "(or copied CNAME name and value from above certificate to aigonewrong.com hosted zone
by adding "create record" just like prior certificate (??? forgot if i did this or aws did it automagicallyh) )

+ "create record" again with "dev*****.aigonewrong.com" and route to same gateway api domain address



+ go to route53, and goto hostedzones->CreateRecord, follow below to create route to api gateway ( use simple record )
```
www.aigonewrong.com
Record type: A 
Route to: api gateway us-east-1 xxxxxxxx.execute-api...
```
https://hackernoon.com/how-to-setup-subdomain-for-aws-api-gateway-d526a9fd6722
https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-to-api-gateway.html



once lambda deploy via zappa, goto coresponding apigateway: https://us-east-1.console.aws.amazon.com/apigateway
+ click `Custom domain names`
+ create domainname `www.aigonewrong.com` and associate it with the just created ACM certificate ( cert may take 10 min to get created/approved )
+ once created, use `Configure API mappings` to map to lambda


```
