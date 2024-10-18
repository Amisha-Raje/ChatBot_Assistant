import requests
import json

cookies = {
    'user.env.type': 'ENTERPRISE',
    'AWSALB': 'xVo0K7eIKmgk5YAHETX9djeOIf2tlZJau6SUm1MGOXy0%2Bz9dlNUme9vwNJO5AhOu2kxFKZnan8VgJTleDk2kzNlWdzCC6ZG3V%2Ff1Gc1T%2Fn0OAUw5DcggiplgNF7H',
    'AWSALBCORS': 'xVo0K7eIKmgk5YAHETX9djeOIf2tlZJau6SUm1MGOXy0%2Bz9dlNUme9vwNJO5AhOu2kxFKZnan8VgJTleDk2kzNlWdzCC6ZG3V%2Ff1Gc1T%2Fn0OAUw5DcggiplgNF7H',
}

headers = {
    'Content-type': 'application/json',
    # 'Cookie': 'user.env.type=ENTERPRISE; AWSALB=xVo0K7eIKmgk5YAHETX9djeOIf2tlZJau6SUm1MGOXy0%2Bz9dlNUme9vwNJO5AhOu2kxFKZnan8VgJTleDk2kzNlWdzCC6ZG3V%2Ff1Gc1T%2Fn0OAUw5DcggiplgNF7H; AWSALBCORS=xVo0K7eIKmgk5YAHETX9djeOIf2tlZJau6SUm1MGOXy0%2Bz9dlNUme9vwNJO5AhOu2kxFKZnan8VgJTleDk2kzNlWdzCC6ZG3V%2Ff1Gc1T%2Fn0OAUw5DcggiplgNF7H',
}

# context = """How to setup Dedicated Webui server :::tip Sample PR : Dedicated Webui setup ::: Adding new deployment for webui in helm-charts Add a new release for the dedicated webui. For creating values.yaml, you can refer to any existing release and modify the params according to your requirements. Next, we need to specify the endpoints our dedicated server will be using. There are 3 endpoints that we will need to specify spaceDomain This will be the url we will finally use to access the dedicated webui server. For eg. space.sprinklr.com webuiDomain For webuiDomain, we follow the convention of appending -app at the end of the domain of spaceDomain endpoint. So, for above example webuiDomain will be space-app.sprinklr.com internalDomain For internalDomain, we follow the convention of prepending api- at the start and appending -int at the end of the domain of spaceDomain endpoint. So, for above example webuiDomain will be api-space-int.sprinklr.com

# query : { "config.type": "PARTNER_SURVEY_CONFIGURATION"} For reference, use config for qa6 partner 66000000 Ref ticket - ITOPS-656404 CACHE_REFRESH - PARTNER_LEVEL_CONFIG Deployments In qa6 and prod0, we have setup dedicated webui, spr-main and api servers to block any interference from changes done by other projects. Dedicated servers will rarely be needed in production envs, but just adding the references here Helm PR for setup - https://prod-gitlab.sprinklr.com/sprinklr-k8s/helm-charts/-/merge_requests/18414 Apart from this, you can refer to webui setup wiki For the api server, we’ll need to expose a CNAME on the external lb so that it is accessible from the surveys-app Apart from the above deployments, we also surveys-app deployment which is a MUST. Helm PR for setup - https://prod-gitlab.sprinklr.com/sprinklr-k8s/helm-charts/-/merge_requests/18925 We also need to expose it’s CNAME Also need to deploy lst-task-ms and inuition microservice Survey Templates

# ::: Deployment Ask noc to do the deployment on both the releases from the latest full build revision. Ask noc to expose the CNAME for all the 3 endpoints we specified and point them to the k8s cluster for the deployments. Once deployment is successful, do proper sanity on the new server. Partner Domain Details configuration By default spr-main-web server hits init-metadata call to the main webui server. So even if we expose a new webui-server and expose a DP (UIUtils.java), it will not reflect for UI. This is because UI is fetching DPs from main webui server which may not have our DPs. Inorder to instruct UI to hit init-metadata to our dedicated webui server, we need to add a PartnerDomainDetails configuration to GLOBAL_SYSTEM_CONFIG mongo like below"""

# question = "How can a Dedicated Webui server be set up?"
data = {
    "model": "gpt-3.5-turbo",
    "provider": "AZURE_OPEN_AI",
    "messages":[
        {"role": "system", "content": """You are an assistant for question-answering tasks. Use ONLY the following pieces of retrieved context to generate answer to the question.If the answer is NOT present in the provided context, say that you don't know. Keep the answer POINTWISE and SHORT (at max 10 sentences). Always say "thanks for asking!" at the end of the answer.."""},
        {"role": "user", "content": """retrieved context:How to setup Dedicated Webui server :::tip Sample PR : Dedicated Webui setup ::: Adding new deployment for webui in helm-charts Add a new release for the dedicated webui. For creating values.yaml, you can refer to any existing release and modify the params according to your requirements. Next, we need to specify the endpoints our dedicated server will be using. There are 3 endpoints that we will need to specify spaceDomain This will be the url we will finally use to access the dedicated webui server. For eg. space.sprinklr.com webuiDomain For webuiDomain, we follow the convention of appending -app at the end of the domain of spaceDomain endpoint. So, for above example webuiDomain will be space-app.sprinklr.com internalDomain For internalDomain, we follow the convention of prepending api- at the start and appending -int at the end of the domain of spaceDomain endpoint. So, for above example webuiDomain will be api-space-int.sprinklr.com

query : { "config.type": "PARTNER_SURVEY_CONFIGURATION"} For reference, use config for qa6 partner 66000000 Ref ticket - ITOPS-656404 CACHE_REFRESH - PARTNER_LEVEL_CONFIG Deployments In qa6 and prod0, we have setup dedicated webui, spr-main and api servers to block any interference from changes done by other projects. Dedicated servers will rarely be needed in production envs, but just adding the references here Helm PR for setup - https://prod-gitlab.sprinklr.com/sprinklr-k8s/helm-charts/-/merge_requests/18414 Apart from this, you can refer to webui setup wiki For the api server, we’ll need to expose a CNAME on the external lb so that it is accessible from the surveys-app Apart from the above deployments, we also surveys-app deployment which is a MUST. Helm PR for setup - https://prod-gitlab.sprinklr.com/sprinklr-k8s/helm-charts/-/merge_requests/18925 We also need to expose it’s CNAME Also need to deploy lst-task-ms and inuition microservice Survey Templates

::: Deployment Ask noc to do the deployment on both the releases from the latest full build revision. Ask noc to expose the CNAME for all the 3 endpoints we specified and point them to the k8s cluster for the deployments. Once deployment is successful, do proper sanity on the new server. Partner Domain Details configuration By default spr-main-web server hits init-metadata call to the main webui server. So even if we expose a new webui-server and expose a DP (UIUtils.java), it will not reflect for UI. This is because UI is fetching DPs from main webui server which may not have our DPs. Inorder to instruct UI to hit init-metadata to our dedicated webui server, we need to add a PartnerDomainDetails configuration to GLOBAL_SYSTEM_CONFIG mongo like below \n\n question:How can a Dedicated Webui server be set up?"""}
    ],
    "temperature": 0.0,
    # "n": 1,
    "stream": False,
    "user": "test:5_108_1312",
    "max_tokens": 500,
    "client_identifier": "backend-insights-dev"
}

response = requests.post(
    'https://prod0-intuitionx-llm-router.sprinklr.com/chat-completion',
    cookies=cookies,
    headers=headers,
    data=json.dumps(data),
)
response_dict = json.loads(response.content)

# print(response.content)
print(response_dict['choices'][0]['message']['content'])