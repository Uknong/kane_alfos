import sys
import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap


def imread_unicode(path):
    with open(path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


class Worker(QThread):
    image_found = pyqtSignal(tuple)
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.template = None
        self.running = False
        self.w, self.h = 0, 0

    def set_template(self, path):
        self.template = imread_unicode(path)
        if self.template is not None:
            self.w, self.h = self.template.shape[::-1]
            self.log_signal.emit(f"템플릿 이미지 로드 성공: {path}")
        else:
            self.log_signal.emit(f"템플릿 이미지 로드 실패: {path}")

    def run(self):
        self.running = True
        base_threshold = 0.65
        self.log_signal.emit("실시간 감지 시작")

        scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                  1.1, 1.2, 1.3, 1.5, 1.7, 2.0]

        while self.running:
            screen = ImageGrab.grab()
            screen_np = np.array(screen)
            frame = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            h_frame, w_frame = gray_frame.shape[:2]

            if self.template is not None:
                found = False
                for scale in scales:
                    w_scaled = int(self.w * scale)
                    h_scaled = int(self.h * scale)

                    # 캡처 화면보다 템플릿 크기가 크면 매칭하지 않음
                    if w_scaled < 10 or h_scaled < 10:
                        continue
                    if w_scaled > w_frame or h_scaled > h_frame:
                        continue

                    resized_template = cv2.resize(self.template, (w_scaled, h_scaled))

                    res = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= base_threshold)

                    for pt in zip(*loc[::-1]):
                        center_x = pt[0] + w_scaled // 2
                        center_y = pt[1] + h_scaled // 2
                        self.image_found.emit((center_x, center_y))
                        self.log_signal.emit(f"이미지 발견 좌표: ({center_x}, {center_y}), 스케일: {scale:.2f}")
                        found = True
                        time.sleep(2)
                        break

                    if found:
                        break

                if not found:
                    self.log_signal.emit("이미지 미발견")

            time.sleep(0.1)

        self.log_signal.emit("실시간 감지 종료")

    def stop(self):
        self.running = False
        self.wait()


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('헉 스키비야!')
        self.resize(320, 350)

        layout = QVBoxLayout()

        self.label = QLabel('이미지를 등록해달라맨이야  <br><br>*beta ver_0.318')
        self.label.setFixedSize(300, 200)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.label)

        btn_load = QPushButton('이미지 등록조이고')
        btn_load.clicked.connect(self.load_image)
        layout.addWidget(btn_load)

        btn_start = QPushButton('자 자동스킵 시작')
        btn_start.clicked.connect(self.start_worker)
        layout.addWidget(btn_start)

        btn_stop = QPushButton('스킵은 안돼요ㅎㅎ')
        btn_stop.clicked.connect(self.stop_worker)
        layout.addWidget(btn_stop)

        self.log_label = QLabel('')
        self.log_label.setFixedHeight(80)
        self.log_label.setStyleSheet("border: 1px solid lightgray; padding: 5px;")
        self.log_label.setWordWrap(True)
        layout.addWidget(self.log_label)

        self.setLayout(layout)

        self.worker = Worker()
        self.worker.image_found.connect(self.handle_image_found)
        self.worker.log_signal.connect(self.show_log)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '이미지 선택', '', 'Image files (*.png *.jpg *.bmp)')
        if fname:
            self.worker.set_template(fname)
            pixmap = QPixmap(fname).scaled(
                self.label.width(), self.label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)

    def start_worker(self):
        if not self.worker.isRunning():
            self.worker.start()
            self.show_log("자동화 시작됨")

    def stop_worker(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.show_log("자동화 중지됨")

    def handle_image_found(self, pos):
        self.show_log(f'이미지 발견, 마우스 이동: {pos}')
        pyautogui.moveTo(pos[0], pos[1], duration=0.2)
        # pyautogui.click()  # 클릭 활성화하려면 주석 해제

    def show_log(self, text):
        self.log_label.setText(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec_())
