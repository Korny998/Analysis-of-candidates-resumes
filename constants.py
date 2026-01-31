from pathlib import Path


AGE_CLASS = [18, 25, 32, 39, 46, 53, 60]

CITY_CLASS = {
    'Московская-область': 0,
    'г-Москва': 0,
    'Ленинградская-область': 1,
    'г-Санкт-Петербург': 1,
    'Новосибирская-область': 2,
    'Свердловская-область': 2,
    'Татарстан-республика': 2,
    'Нижегородская-область': 2,
    'Красноярский-край': 2,
    'Челябинская-область': 2,
    'Самарская-область': 2,
    'Башкортостан-республика': 2,
    'Ростовская-область': 2,
    'Краснодарский-край': 2,
    'Омская-область': 2,
    'Воронежская-область': 2,
    'Пермский-край': 2,
    'Волгоградская-область': 2,
    'Прочие-города': 3
}

CSV_PATH = Path('cv_100000.csv')

DROP = [
    'id', 'candidateId', 'stateRegionCode', 'locality',
    'birthday', 'gender', 'dateCreate', 'dateModify',
    'publishedDate', 'academicDegree', 'worldskills',
    'worldskillsInspectionStatus', 'abilympicsInspectionStatus',
    'abilympicsParticipation', 'volunteersInspectionStatus',
    'volunteersParticipation', 'driveLicenses', 'professionsList',
    'otherCertificates', 'narkCertificate', 'narkInspectionStatus',
    'codeExternalSystem', 'country', 'additionalEducationList',
    'hardSkills', 'softSkills', 'retrainingCapability', 'businessTrip',
    'languageKnowledge', 'relocation', 'innerInfo'
]

EMPLOYMENT_CLASS = {
    'Стажировка': 0,
    'Временная': 1,
    'Сезонная': 2,
    'Частичная-занятость': 3,
    'Удаленная': 4,
    'Полная-занятость': 5
}

EXPERIENCE_CLASS = [1, 3, 5, 7, 10, 15]

FILTERS = '!"«»#$№%&()*+,-–—./:;<=>?@[\\]^_`{|}~\t\n\xa0'

SCHEDULE_CLASS = {
    'Сменный-график': 0,
    'Ненормированный-рабочий-день': 1,
    'Вахтовый-метод': 2,
    'Гибкий-график': 3,
    'Неполный-рабочий-день': 4,
    'Полный-рабочий-день': 5,
}

NUM_WORDS = 3000

URL: str = 'https://storage.yandexcloud.net/academy.ai/cv_100000.csv'
