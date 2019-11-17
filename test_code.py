model_url = ''.join(['https://gist.github.com/zhreshold/',
                     'bcda4716699ac97ea44f791c24310193/raw/',
                     '93672b029103648953c4e5ad3ac3aadf346a4cdc/',
                     'super_resolution_0.2.onnx'])
model_path = download_testdata(model_url, 'super_resolution.onnx', module='onnx')