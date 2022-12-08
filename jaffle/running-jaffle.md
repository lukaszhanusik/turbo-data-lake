# AWS Localstack

## Make bucket in S3

```sh
localstack start
awslocal s3 mb s3://datalake
```

```
gitpod /workspace/empty (main) $ awslocal s3 mb s3://datalake
make_bucket: datalake
gitpod /workspace/empty (main) $ 
```

## Testing 

```
gitpod /workspace/empty (main) $ cd jaffle
gitpod /workspace/empty/jaffle (main) $ pytest -s jaffle_tests
```

```
===================================================================================== test session starts =====================================================================================
platform linux -- Python 3.8.13, pytest-7.2.0, pluggy-1.0.0
rootdir: /workspace/empty/jaffle
plugins: anyio-3.6.2, pylama-8.4.1
collecting 1 item                                                                                                                                                             collected 1 item                                                                                                                                                                              

jaffle_tests/test_assets.py .

====================================================================================== 1 passed in 3.36s ======================================================================================
```

