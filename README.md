# cortex-npdb

This is the CORTEX-NPDB model project.

The project is set up as a Docker image. This document describes the steps you need to follow in order to get the project up and running.

The old README can be found [here](docs/README-old.md).

## Get the project

1. Get an AWS IAM user with its corresponding credentials.
2. [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html).
3. [Configure the CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) with your credentials.
4. Install and run [docker](https://docs.docker.com/get-docker/).
5. Authenticate [AWS ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html):
  - Retrieve an authentication token and authenticate your Docker client to your registry.
  ```
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 370076501934.dkr.ecr.us-east-1.amazonaws.com
  ```
6. Go to the **repositories** section in the AWS ECR dashboard in the AWS Console.
7. Go to the cortex-npdb/webservice repository.
8. Get the latest image tag ID.
9. Pull the latest image.
```
  docker pull 370076501934.dkr.ecr.us-east-1.amazonaws.com/cortex-npdb/webservice:<image id>
```

## Run the API

```
docker run -it -p 8080:8080 -d <image id>
```

## Run the automated tests
```
docker exec -it <image id> python3 -m unittest test/main_npdb_cover_beta_test.py
```
