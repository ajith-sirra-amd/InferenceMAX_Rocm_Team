import sys
import requests

repository = sys.argv[1]
auth_url = f'https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull'
token = requests.get(auth_url).json()['token']

tags_url = f'https://registry-1.docker.io/v2/{repository}/tags/list'
resp = requests.get(tags_url, headers={'Authorization': f'Bearer {token}'})
resp.raise_for_status()

vllm_tags = resp.json()['tags']
valid_tags = [tag for tag in vllm_tags if tag.startswith(sys.argv[2]) and 'rc' not in tag]


def make_key_cuda(tag):
    '''
    Tag format: vX.Y.Z(.W)
    X, Y, Z are numbers
    W can be a number or string postN (N is a number)
    '''
    vals = tag[1:].split('.')
    post = vals[3] if len(vals) == 4 else '0'
    key = (int(vals[0]), int(vals[1]), int(vals[2]), post)
    return key

def make_key_rocm(tag):
    *_, date = tag.split('_')
    try:
        key = int(date)
    except:
        key = -1
    return key

if repository == 'vllm/vllm-openai':
    make_key_fn = make_key_cuda
elif repository == 'rocm/vllm':
    make_key_fn = make_key_rocm
else:
    raise ValueError(f'Invalid repo {repository}')

tag = max(valid_tags, key=make_key_fn)
print(f'{repository}:{tag}')
