#统计大小写字符的个数
def AdjustTuple(string):
    first=second=0
    for c in string:
        if c.isupper():
            first += 1
        elif c.islower():
            second += 1
    return (first,second)

#主函数
if __name__ == '__main__':
    string=input("请输入一行字符串:")
    print(AdjustTuple(string))