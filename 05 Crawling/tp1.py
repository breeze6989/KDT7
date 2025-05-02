from selenium import webdriver
from selenium.webdriver.common.by import By

def link_crawl(driver:webdriver.Chrome):
    array= []
    f = open(r"C:\data\jobkorea_link.txt",mode="w",encoding="utf-8")
    for i in range(1,52):
        driver.get("https://www.jobkorea.co.kr/starter/PassAssay?FavorCo_Stat=0&Pass_An_Stat=0&OrderBy=0&EduType=0&WorkType=0&schPart=10031&isSaved=1&Page="+str(i))
        paper_list = driver.find_element(By.XPATH, '//*[@id="container"]/div[2]/div[5]/ul')
        driver.implicitly_wait(3)
        urls = paper_list.find_elements(By.TAG_NAME,'a')
        for url in urls:
            if 'selfintroduction' in url.get_attribute('href'):
                pass
            else:
                array.append(url.get_attribute('href'))
    array = list(set(array))
    for content in array:
        f.write(content+'\n')
    f.close()

def login_protocol(driver:webdriver.Chrome): # 로그인해야지 로그인창때문에 크롤링 멈추는거 막을 수 있음
    driver.get("https://www.jobkorea.co.kr/")
    driver.find_element(By.XPATH,'//*[@id="devMyPage"]/ul/li[1]/a').click()
    driver.find_element(By.NAME,"M_ID").send_keys("yoon660033")
    driver.find_element(By.NAME,"M_PWD").send_keys("!@ajajfldydnjs9")
    driver.find_element(By.XPATH,'//*[@id="login-form"]/fieldset/section[3]/button').click()
    driver.implicitly_wait(3)
    # driver.find_element(By.ID,"closeIncompleteResume")
    # driver.implicitly_wait(3)
    print("login success")
spec=[]
def self_introduction_crawl(driver:webdriver.Chrome,file_url):
    print("current URL : "+ file_url)
    driver.get(file_url)
    user_info = driver.find_element(By.XPATH,'//*[@id="container"]/div[2]/div[1]/div[1]/h2')
    company = user_info.find_element(By.TAG_NAME,'a')
    print(company.text) # 지원회사
    season= user_info.find_element(By.TAG_NAME,'em')
    print(season.text) # 지원시기
    
    specification=driver.find_element(By.CLASS_NAME,'specLists')
    spec_array = specification.text.split('\n')
    spec.append(spec_array)
    print(spec_array[:-2]) #스펙
    paper = driver.find_element(By.CLASS_NAME,"qnaLists")
    questions = paper.find_elements(By.TAG_NAME,'dt')
    print("question")
    for index in questions:
        question = index.find_element(By.CLASS_NAME,'tx')
        if question.text=="":
            index.find_element(By.TAG_NAME,'button').click()
            question = index.find_element(By.CLASS_NAME,'tx')
            print(question.text)
        else:
            print(question.text) # 자소서 질문 모아놓은 리스트
    driver.implicitly_wait(3)
    answers = paper.find_elements(By.TAG_NAME,'dd')
    driver.implicitly_wait(3)
    print('answer')
    for index in range(len(answers)):
        answer =answers[index].find_element(By.CLASS_NAME,'tx')
        if answer.text == "":
            questions[index].find_element(By.TAG_NAME,'button').click()
            answer =answers[index].find_element(By.CLASS_NAME,'tx')
        print(answer.text) # 자소서 답변 모아놓은 리스트

def save_spec_to_file():
    file_path = r"C:\data\jobkorea_spec.txt"
    
    # 파일 쓰기 모드로 열기
    with open(file_path, "w", encoding="utf-8") as f:
        for spec_item in spec:
            f.write(", ".join(spec_item) + "\n")  # 리스트를 쉼표로 구분하여 저장

    print(f"스펙 데이터가 {file_path} 파일에 저장되었습니다.")
         
file = open(r'C:\data\jobkorea_link.txt','r')
driver = webdriver.Chrome()
# link_crawl(driver=driver)
login_protocol(driver=driver)
while True: 
    file_url = file.readline()
    if file_url == "":
        break
    self_introduction_crawl(driver=driver,file_url=file_url)

save_spec_to_file() 

