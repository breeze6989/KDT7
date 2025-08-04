-- 사용할 데이터베이스 생성 (이미 있다면 생략 가능)
CREATE DATABASE IF NOT EXISTS fire_dashboard_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 생성한 데이터베이스 사용
USE fire_dashboard_db;

-- 사용자 정보를 저장할 'users' 테이블 생성
CREATE TABLE IF NOT EXISTS `users` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `username` VARCHAR(50) NOT NULL UNIQUE,
    `password` VARCHAR(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 프론트엔드 코드에 명시된 테스트 계정 삽입
-- 실제 운영 환경에서는 반드시 비밀번호를 해시하여 저장해야 합니다.
INSERT INTO `users` (username, password) VALUES ('admin', 'adminpass')
ON DUPLICATE KEY UPDATE password='adminpass';