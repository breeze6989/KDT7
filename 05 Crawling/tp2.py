from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException

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

def login_protocol(driver:webdriver.Chrome):
    driver.get("https://www.jobkorea.co.kr/")
    driver.find_element(By.XPATH,'//*[@id="devMyPage"]/ul/li[1]/a').click()
    driver.find_element(By.NAME,"M_ID").send_keys("yoon660033")
    driver.find_element(By.NAME,"M_PWD").send_keys("!@ajajfldydnjs9")
    driver.find_element(By.XPATH,'//*[@id="login-form"]/fieldset/section[3]/button').click()
    driver.implicitly_wait(3)
    print("login success")

spec=[]

def self_introduction_crawl(driver:webdriver.Chrome, file_url):
    print("current URL : " + file_url)
    try:
        driver.get(file_url)
        
        user_info = driver.find_element(By.XPATH,'//*[@id="container"]/div[2]/div[1]/div[1]/h2')
        company = user_info.find_element(By.TAG_NAME,'a')
        print(company.text) # 지원회사
        season= user_info.find_element(By.TAG_NAME,'em')
        print(season.text) # 지원시기

        specification = driver.find_element(By.CLASS_NAME,'specLists')
        spec_array = specification.text.split('\n')
        spec.append(spec_array)
        print(spec_array[:-2]) #스펙

        paper = driver.find_element(By.CLASS_NAME,"qnaLists")
        questions = paper.find_elements(By.TAG_NAME,'dt')
        print("question")
        for index in questions:
            question = index.find_element(By.CLASS_NAME,'tx')
            if question.text == "":
                index.find_element(By.TAG_NAME,'button').click()
                question = index.find_element(By.CLASS_NAME,'tx')
            print(question.text)

        answers = paper.find_elements(By.TAG_NAME,'dd')
        print('answer')
        for index in range(len(answers)):
            answer = answers[index].find_element(By.CLASS_NAME,'tx')
            if answer.text == "":
                questions[index].find_element(By.TAG_NAME,'button').click()
                answer = answers[index].find_element(By.CLASS_NAME,'tx')
            print(answer.text)

    except NoSuchElementException as e:
        print(f"[ERROR] 자소서가 존재하지 않는 것으로 보입니다. 스킵합니다. 에러 상세: {e}")
        return
    except TimeoutException as e:
        print(f"[ERROR] 페이지 로딩이 오래 걸려 타임아웃이 발생했습니다. 스킵합니다. 에러 상세: {e}")
        return
    except Exception as e:
        print(f"[ERROR] 알 수 없는 예외 발생: {e}")
        return


def save_spec_to_file():
    file_path = r"C:\data\jobkorea_spec.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        for spec_item in spec:
            f.write(", ".join(spec_item) + "\n")
    print(f"스펙 데이터가 {file_path} 파일에 저장되었습니다.")

file = open(r'C:\data\jobkorea_link.txt','r', encoding='utf-8')
driver = webdriver.Chrome()
# link_crawl(driver=driver)  # 이미 링크를 수집하셨다면 주석처리
login_protocol(driver=driver)

while True:
    file_url = file.readline()
    if file_url == "":
        break
    # 파일에 개행 포함되어 있으면 strip() 해주는 게 안전
    file_url = file_url.strip()
    self_introduction_crawl(driver=driver,file_url=file_url)

save_spec_to_file()
file.close()
