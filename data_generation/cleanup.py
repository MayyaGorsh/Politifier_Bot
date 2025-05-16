res = []

with open('before_cleanup.txt', encoding='utf-8') as file:
    lines = file.readlines()

if 'ChatGPT\n' in lines:
    raise ValueError('почисти строки')

if len(lines) < 295:
    for i in range(len(lines)):
        if lines[i][0] == '"':
            res.append(lines[i].strip().replace('"', ''))
    print('грубые', len(res))
else:
    for i in range(1, len(lines), 3):
        res.append(lines[i].strip().replace('"', ''))
    print('вежливые')


with open('after_cleanup.txt', 'w', encoding='utf-8') as file:
    for i in res:
        print(i, file=file)