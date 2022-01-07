import valohai
file=valohai.outputs().path('message.txt')
with open(file, 'w') as f:
    f.write('hello\n') 