from django.shortcuts import render

from static.preprocessing.prep import get_test_result


# Create your views here.
def index(request):
    if request.method == "GET":
        return render(request, "makecrimeanalysis/index.html")
    else:
        result = get_test_result(request)
        return render(request, "makecrimeanalysis/index.html", {
            'isOk' : True,
            'code' : result
        })
