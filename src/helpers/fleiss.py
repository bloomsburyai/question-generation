"""
fleiss.py by Marco Lui, Dec 2010

Based on
  http://en.wikipedia.org/wiki/Fleiss'_kappa
and
  Cardillo G. (2007) Fleisses kappa: compute the Fleiss'es kappa for multiple raters.
  http://www.mathworks.com/matlabcentral/fileexchange/15426
"""
import numpy
# from scipy.special import erfc

def fleiss(data):
  if not len(data.shape) == 2:
    raise(ValueError, 'input must be 2-dimensional array')
  if not issubclass(data.dtype.type, numpy.integer):
    raise(TypeError, 'expected integer type')
  if not numpy.isfinite(data).all():
    raise(ValueError, 'all data must be finite')

  raters = data.sum(axis=1)
  if (raters - raters.max()).any():
    raise(ValueError, 'inconsistent number of raters')

  num_raters = raters[0]
  num_subjects, num_category = data.shape
  total_ratings = num_subjects * num_raters

  pj = data.sum(axis=0) / float(total_ratings)
  pi = ((data * data).sum(axis=1) - num_raters).astype(float) / ( num_raters * (num_raters-1) )
  pbar = pi.sum() / num_subjects
  pebar = (pj * pj).sum()

  kappa = (pbar - pebar) / (1 - pebar)
  return kappa

# def fleiss(data):
#   if not len(data.shape) == 2:
#     raise ValueError, 'input must be 2-dimensional array'
#   if not issubclass(data.dtype.type, numpy.integer):
#     raise TypeError, 'expected integer type'
#   if not numpy.isfinite(data).all():
#     raise ValueError, 'all data must be finite'
#
#   raters = data.sum(axis=1)
#   if (raters - raters.max()).any():
#     raise ValueError, 'inconsistent number of raters'
#
#   n, num_category = data.shape
#   # m=sum(x(1,:)); %raters
#   m = data[0].sum(axis=0)
#
#   # a=n*m;
#   a = n * m
#
#   # pj=(sum(x)./(a)); %overall proportion of ratings in category j
#   pj = data.sum(axis=0) / float(a)
#
#   # b=pj.*(1-pj);
#   b = pj * (1-pj)
#
#   # c=a*(m-1);
#   c = a * (m-1)
#
#   # d=sum(b);
#   d = numpy.sum(b, axis=0)
#
#   # kj=1-(sum((x.*(m-x)))./(c.*b)); %the value of kappa for the j-th category
#   kj = 1 - ( (data * (m-data)).sum(axis=0)  / (c*b) )
#
#   # sekj=realsqrt(2/c); %kj standar error
#   sekj = numpy.sqrt(2/c)
#
#   # zkj=kj./sekj;
#   zkj = kj / sekj
#
#   # pkj=(1-0.5*erfc(-abs(zkj)/realsqrt(2)))*2;
#   pkj = (1 - 0.5*erfc(-numpy.abs(zkj) / numpy.sqrt(2))) * 2
#
#   # k=sum(b.*kj)/d; %Fleiss'es (overall) kappa
#   k = (b*kj).sum(axis=0) / d
#
#   # sek=realsqrt(2*(d^2-sum(b.*(1-2.*pj))))/sum(b.*realsqrt(c)); %kappa standard error
#   sek = numpy.sqrt( 2*(d*d-(b*(1-2*pj)).sum(axis=0)) ) / (b * numpy.sqrt(c)).sum(axis=0)
#
#   # ci=k+([-1 1].*(abs(0.5*erfc(-alpha/2/realsqrt(2)))*sek)); %k confidence interval
#   # omitted as we are not working out the ci
#
#   # z=k/sek; %normalized kappa
#   z = k/sek
#
#   # p=(1-0.5*erfc(-abs(z)/realsqrt(2)))*2;
#   p = (1 - 0.5*erfc(-numpy.abs(z) / numpy.sqrt(2))) * 2
#   return k, p

if __name__ == "__main__":
  data = numpy.array([
    [0,0,0,0,14],
    [0,2,6,4,2],
    [0,0,3,5,6],
    [0,3,9,2,0],
    [2,2,8,1,1],
    [7,7,0,0,0],
    [3,2,6,3,0],
    [2,5,3,2,2],
    [6,5,2,1,0],
    [0,2,2,3,7],
    ])

  print(fleiss(data))
