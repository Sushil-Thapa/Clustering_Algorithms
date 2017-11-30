import time,resource,os

# kmeans=[]
# single_pass = []
# ktree = []
path, dirs, files = os.walk("complexities").next()
file_suffix = max(0,len(files)//3 - 1)
mode = 'w'
def timeit(method):
    algorithm = method.__name__
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        tt = time.time() -ts
        # print '%r (%r, %r) %2.2f sec' % (algorithm, args, kw, tt)
        print '%r for %r clusters in %2.2f sec' % (algorithm,args[0], tt)
        # print("Max Ram Usage: %.2f MB.\n" % (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024)) #TODO make function f and use mem_usage = memory_usage(f) an its max

        # globals().get(algorithm, []).append((args[0],tt)) #
        # saveFile(algorithm,tt)
        return result
    return timed

def saveFile(algorithm,t):
    global mode
    fname = 'complexities/'+str(algorithm)+'_'+str(file_suffix)+'.csv'
    with open(fname, mode) as myfile:
        mode = 'a'
        myfile.write(str('1000000')+','+str(round(float(t),3))+'\n')
        myfile.close()
    print(fname+' saved.')
