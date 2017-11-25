import time,resource

# kmeans=[]
# single_pass = []
# ktree = []
mode = 'w'
def timeit(method):
    algorithm = method.__name__
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        tt = te-ts
        # print '%r (%r, %r) %2.2f sec' % (algorithm, args, kw, tt)
        print '%r %r datasets in %2.2f sec' % (algorithm,args[1], tt)
        print("Max Ram Usage: %.2f MB.\n" % (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024)) #TODO make function f and use mem_usage = memory_usage(f) an its max

        # globals().get(algorithm, []).append((args[0],tt)) #
        saveFile(algorithm,args[0],args[1],tt)
        return result

    return timed

def saveFile(algorithm,n,frac,t):
    global mode
    with open('complexities/'+algorithm+'.csv', mode) as myfile:
        mode = 'a'
        myfile.write(str(frac)+','+str(round(float(t),3))+'\n')
        myfile.close()
