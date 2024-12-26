import pygame
import random
import numpy as np

from plot_script import plot_result

# Настройки игры
WIDTH, HEIGHT = 800, 600
FPS = 60
NUM_ACTIONS = 4  # Влево, Вправо, Вверх, Вниз

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

rewards = []
episods = 20


class Item:
    def __init__(self):
        self.image = pygame.Surface((50, 50))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, WIDTH - 50)
        self.rect.y = random.randint(-100, -40)
        self.speed = random.randint(1, 5)

    def update(self, collision_detected):
        self.rect.y += self.speed
        if self.rect.y > HEIGHT or collision_detected:
            self.rect.y = random.randint(-100, -40)
            self.rect.x = random.randint(0, WIDTH - 50)





class Bucket:
    def __init__(self):
        self.image = pygame.Surface((50, 50))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT))
        self.speed = 5

    def update(self, action):
        if action == 0:  # Влево
            self.rect.x -= self.speed
        elif action == 1:  # Вправо
            self.rect.x += self.speed


        # Ограничения по экрану
        if self.rect.x < 0:
            self.rect.x = 0
        elif self.rect.x > WIDTH - self.rect.width:
            self.rect.x = WIDTH - self.rect.width
        if self.rect.y < 0:
            self.rect.y = 0
        elif self.rect.y > HEIGHT - self.rect.height:
            self.rect.y = HEIGHT - self.rect.height


class QLearningAgent:
    def __init__(self, actions):
        self.q_table = np.zeros((WIDTH // 10, HEIGHT // 10, len(actions)))  # Q-таблица
        self.actions = actions
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # Начальная вероятность выбора случайного действия
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Случайное действие
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Лучшая оценка

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action]
        td_delta = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.learning_rate * td_delta

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    bucket = Bucket()
    items = [Item() for _ in range(5)]
    agent = QLearningAgent(actions=[0, 1])  # Действия: влево, вправо

    running = True
    for e in range(episods):
        sum_of_rewards = 0
        for i in range(1000):
            state = (bucket.rect.x // 10, bucket.rect.y // 10)  # Дискретизация состояния
            action = agent.choose_action(state)
            bucket.update(action)

            collision_detected = False

            # Проверка столкновений
            reward = -1  # Штраф за каждое действие (избегание столкновения)

            # Обновление предметов
            for item in items:
                item.update(False)

            for item in items:
                if bucket.rect.colliderect(item.rect):
                    print("Поймал!")
                    reward = 10  # Награда за поимку предмета
                    collision_detected = True
                    item.update(True)
                    break

            next_state = (bucket.rect.x // 10, bucket.rect.y // 10)

            agent.learn(state, action, reward, next_state)

            # Обновление epsilon для исследования
            agent.update_epsilon()

            # Отображение
            screen.fill(BLACK)
            screen.blit(bucket.image, bucket.rect)
            for item in items:
                screen.blit(item.image, item.rect)

            pygame.display.flip()
            clock.tick(FPS)
            sum_of_rewards += reward
        rewards.append(sum_of_rewards)
        print("Номер эпизода ", e)

    pygame.quit()

    # Построение графика с результатом игры
    params = dict()
    params['name'] = None
    results = dict()
    print(rewards)
    results[params['name']] = rewards
    plot_result(results, direct=True, k=20)
    print(len(rewards))

if __name__ == "__main__":
    main()
