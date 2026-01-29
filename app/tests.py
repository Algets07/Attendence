# from django.test import TestCase

# Create your tests here.



a= [9,7,4,5,3,1,2,6]

for i in range(len(a)):
    for j in range(i+1,len(a)):
        print(f'index: i: {i} and j: {j} compares {a[i]} and {a[j]}')
        if a[i]>a[j]:
            print('\n',a)
            a[i],a[j]=a[j],a[i]
            print(a,'\n\n')


b=a.avg()