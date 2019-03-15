import urllib
import os
from bs4 import BeautifulSoup
import ssl
import re


# TODO:加曲谱录入者

def create_folder(save_path, i=0):
    # 新建文件夹
    if i != 0:
        save_path = save_path + '_' + str(i)
    if os.path.exists(save_path):
        if i == 0:
            save_path = create_folder(save_path, i=i + 1)
        else:
            save_path = create_folder(save_path[:-2], i=i + 1)
    else:
        os.makedirs(save_path)
    return save_path


def get_html_soup(url, no_gbk=False):
    # 返回给定url下的soup实例
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    path, name = os.path.split(url)
    url = path + '/' + urllib.request.quote(name)
    try:
        if no_gbk:
            html = urllib.request.urlopen(url, context=ctx).read()
        else:
            html = urllib.request.urlopen(url, context=ctx).read().decode('gbk').encode('utf-8')
        soup = BeautifulSoup(html, "html.parser")
        return soup

    except(urllib.error.URLError):
        return None


def get_img(url, save_dir):
    if url.find('ShowPhoto') > 0:
        # 过滤需要登录的谱子
        # Filter the scores that need login
        pass
    else:
        soup = get_html_soup(url)
        if soup != None:
            save_path = os.path.join(save_dir, soup.find_all('div')[2].text.replace(u'\xa0', u' '))
            save_path = create_folder(save_path)
            img_urls = re.findall('arrUrl\[[0-9]+\]=\'(\S+)\'', soup.find_all('div')[8].text)  # 一个谱子的所有页
            save_img(img_urls, save_path)


def save_img(img_urls, save_path):
    # 保存图片
    try:
        for url in img_urls:
            filepath, filename = os.path.split(url)
            img_url = filepath + '/' + urllib.request.quote(filename)
            sp = os.path.join(save_path, filename)
            urllib.request.urlretrieve(img_url, sp)
    except(urllib.error.URLError):
        pass


def get_instruments_url(url):
    # 得到所有乐器的谱子地址
    # Get the urls of all instrument
    soup = get_html_soup(url, no_gbk=True)
    return [tag.get('href') for tag in
            soup.find_all('body')[0].find_all(name='div', attrs={"class": "pub2"})[0].find_all('a')]


def get_instrument_score(url, save_path):
    # 得到当前乐器的所有谱子地址
    # Get all the scores in current instrument
    soup = get_html_soup(url)
    instrument_name = soup.find_all('body')[0].find_all(name='div', attrs={"class": "pub1"})[0].find_all('h1')[0].text
    save_path = os.path.join(save_path, instrument_name)
    tag = soup.find_all('body')[0].find_all(name='div', attrs={"class": "pub"})[0].find_all(name='div',
                                                                                            attrs={"class": "c628_o"})[
        0].find_all('a')
    page_url_list = []
    for t in tag:
        if t.string == '2':
            page_two = t['href']
            num = page_two[-6]
            for i in range(1, int(num) + 1):
                page_url_list.append(page_two[:-6] + str(i) + page_two[-5:])
    page_url_list.append(url)
    for page_url in page_url_list:
        # print('Page',page_url)
        get_current_page(page_url, save_path)


def get_current_page(page_url, save_path):
    # 下载当前页（谱子）下的所有图片（所有页）
    # Download all the image score in current page (score)
    soup = get_html_soup(page_url)
    tags = soup.find_all('body')[0].find_all(name='div', attrs={"class": "pub"})[0].find_all(name='div',
                                                                                             attrs={"class": "c628_v"})[
        0].find_all('table')[1].find_all('tr')
    current_page_urls = []
    for tag in tags:
        tag_a = tag.find_all('a')
        if len(tag_a) != 0:
            current_page_urls.append(tag_a[0]['href'])
    for url in current_page_urls:
        pass
        # print('Score',url)
        get_img(url, save_path)


if __name__ == '__main__':
    url = 'http://pu.guqu.net/'
    save_path = '古谱'

    instruments_url_list = get_instruments_url(url)
    for instruments_url in instruments_url_list:
        # print('Instrument:',instruments_url)
        get_instrument_score(instruments_url, save_path)
