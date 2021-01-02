---
layout: post
title: Jekyll 테마 이용해서 블로그 만들기
tags: [tutorial]
math: true
date: 2021-01-02 21:49 
comments : true
---

github + jekyll 테마를 이용해 블로그를 만드는 방법을 정리해보겠습니다.

## 1. Github Repository 생성
---
먼저 https://github.com을 방문하여 가입합니다. 

로그인 후, 우측 상단의 Profile > Your repositories > New를 클릭합니다. 

![](/assets/images\start_blog_2.png)

Repository name 란에 `USERNAME.github.io` (where `USERNAME` is your GitHub username)을 입력하고 Create repository를 클릭합니다.

```terminal
$ git clone https://github.com/USERNAME/USERNAME.github.io.git -b main --single-branch
```
> Github는 2020.10월 이후부터 새로 생성되는 repository는 Default Branch를 master에서 main으로 변경(그 이전은 master) <br>
(참고) (https://dunchi.tistory.com/92)

---
## 2. jekyll 테마 선정

:grey_exclamation: Jekyll(지킬) 
 > 설치형 블로그, Ruby 기반 <br>
 > Markdown을 사용해서 포스트를 작성하면 HTML로 변환 <br>
 > Github pages에 무료 호스팅을 제공하기 때문에 많이 사용 <br>
 > <U> 즉, 전문가들이 만들어 놓은 블로그 테마를 수정해서 사용할 수 있다는 것! </U> <br>
 
[jekyll theme](http://jekyllthemes.org/) 사이트에 들어가서 맘에 드는 테마를 선택한다. <br>

필자는 https://vszhub.github.io/not-pure-poole/dates/ 테마를 선택하였다. (테마에 따라 github 설정 방식이 다를 수 있음) <br>

해당 테마의 github 페이지에 들어가서 fork 또는 download 한다. 
![](/assets/images\start_blog_4.png)

> fork시 github 잔디가 심어지지 않아, download 하여 진행하였다. 

download 받은 zip 파일을 앞에서 만든 github Repository(USERNAME.github.io) 폴더 하위에 압축을 해제한다. 

---
## 3. jekyll 테마 _config.yml 수정 
블로그의 제목, 주소, url 기본 설정 값이 있는 _config.yml 파일을 수정한다. <br>
이 부분은 테마에 따라 설정 방식이 다를 수 있으므로, 테마 제작자의 설명서를 확인해야 합니다(ReadME.md). 

> 기본적으로 수정해야 할 것
- 블로그 title 
- 블로그 URL 
- 블로그 이미지
- 테마 제작자의 설명(ReadME.md)에서 추가하는 플러그인 (**중요**)

![](/assets/images\start_blog_5.png)

---
## 4. git push (업데이트 코드를 git에 업로드)

`git push`는 원격 저장소(remote repository)에 코드 변경분을 업로드하기 위해서 사용하는 Git 명령어 입니다.

```terminal
# 변경한 영역을 모두 stage에 올림
$ git add --all  
```
![](/assets/images\start_blog_6.png)
위와 같은 에러는 OS별 CRLF 차이로 인해 발생하므로, CRLF 옵션을 변경해줘야 한다. <br>
`CRLF`는 새로운 줄(New line)로 바꾸는 방식을 의미한다. 

윈도우에서는 CRLF를 사용하므로 저장소에서 가져올 때에는 LF를 CRLF로 변경하고 저장소로 보낼때는 CRLF를 LF로 변경하도록 true로 설정한다. 

```terminal
$ git config --global core.autocrlf true  
```
```terminal
$ git add --all
$ git commit -m "first changed"
$ git push
```
`USERNAME.github.io` 에 들어가 블로그 화면이 정상적으로 나오는지 확인한다. 

---
## 5. blog post 생성 (새글 작성)

blog post는 USERNAME.github.io > _posts 폴더에 markdown 파일(.md)을 생성하여 만들 수 있다. 

1. markdown 파일 생성

- `YYYY-MM-DD-TITLE.md` 파일 생성 
- 해당 파일을 `USERNAME.github.io > _posts` 폴더에 저장

![](/assets/images\start_blog_7.png)

2. post 내용 작성(markdown) <br>

아래와 같은 형식으로 title , tags, date 를 작성한다(**블로그 테마마다 방식 상이**) 

```markdown
layout: post
title: Jekyll 테마 이용해서 블로그 만들기
tags: [tutorial]
math: true
date: 2021-01-02 21:49 
```
본문 작성(markdown 방식으로)

본문을 작성할 때는 markdown 에디터 사용 <br>

<Visual Studio code - markdown editor 사용 예시>

![](/assets/images\start_blog_8.png)

3. git push 
```terminal
$ git add --all
$ git commit -m "first changed"
$ git pus

```

(markdown 사용법 참고)
https://gist.github.com/ihoneymon/652be052a0727ad59601

---

### 결론
- jekyll 초보자일 경우, jekyll 테마 선정시, 테마에서 제공하는 기능이 복잡할 경우에는 비추 <br>
- 가급적이면 처음 블로그를 만드는 분이시라면, 테마 제공 기능이 비교적 simple 한 걸 추천! <br> 
  (github page build 에러가 뜨면 수정하기 여러움)
- **markdown 사용법 숙지 필요** 
- 테마 제작자의 테마 사용법(Readme.md) 꼼꼼히 보기




