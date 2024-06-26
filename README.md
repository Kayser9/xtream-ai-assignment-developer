# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run

In order to have a an prompt snapshot of my project, including the step by step evolution, I decided to generate a folder structure broken down it following the given challenges.

Within challenge1 and challeng2 two github actions(workflow) have been deployed to train the models storing the internal training __Logs__ and the __Models Artifacts__ remotely.

Since I hadn't a centralized remote storage system, I used multiple docker containers binded with two name volumes to ensure data persistence:

- {dock_comp_name_prefix}_logs -> /app/Logs : used to store the training session logs
- {dock_comp_name_prefix}_models -> /app/Models : used to store the model artifacts (Trained models in pkl format) and a "report.json" file containing all the model information

By default, docker-compose will create also the two named volumes and the network.

### Challenge 1
__Run the linear model pipeline__:
- __docker-compose run --build pipeline__ : build image -> run container (attach mode, i.e. output) -> End.
- __docker-compose run --build pipeline /bin/bash__ : build image -> run container interactively (i.e input and output).
--  Within the container's terminal, to train the model: __python main.py__

- __docker compose down -v__ : Destroy container, network and volumes

### Challenge 2

- __docker-compose run --build pipeline__ : build image -> run container (attach mode, i.e. output) -> End.
-- By default, the container will train the __xgboost__ model. 
- __docker-compose run --build pipeline /bin/bash__ : build image -> run container interactively (i.e input and output).
--  Within the container's terminal, to train the model: __python main.py <model_type>__ (linear/xgboost)

- __docker compose down -v__ : Destroy container, network and volumes


### Challenge 3
I have deployed two container, the api server and the pipeline one. Both containers share the volumes, however, the api one has just read only permissions.

To test the api (mapped on port 8080:8080), is possible to use the __call.rest__ file or by searching in the browser __localhost:8080/docs__.

Two enpoint have been deployed and are both visible within the __call_rest__ file or directly in the __api.py__ script. Since both services start simultaneously, the Api won't find any available models until at least one model is trained.

Within the api, the best performing model will be picked and exposed to the enpoints.


#### Run both services together
- __docker_compose up --build__: Run both containers in attach mode; append the -d flag to run them in background. The model training takes about 1 minute. After that it will be exposed by the api.

#### Run just one service/container
- __docker-compose run --build pipeline__ : build image -> run container (attach mode, i.e. output) -> End.
-- By default, the container will train the __xgboost__ model. 

- __docker-compose run --build -it --rm pipeline /bin/bash__ : build image -> run container interactively (i.e input and output).
--  Within the container's terminal, to train the model: __python main.py <model_type>__ (linear/xgboost)

- __docker-compose run -it --build --rm api__  : build image -> run container (attach mode) -> Start the server.

- __docker-compose run --build --rm api /bin/bash__ : build image -> run container interactively (i.e input and output). To start the server: __uvicorn api:app --host 0.0.0.0 --port=8080. 


- __docker compose down -v__ : Destroy container, network and volumes


All data are persistent and available within the named volumes. However once they are destroyed, all the data they were storing will be lost.



## Thank you very much for the opportunity,


### Luca Checchin

