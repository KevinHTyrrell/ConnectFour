from Bots.BaseBot import BaseBot
from Game import Game

if __name__ == "__main__":
    session = Game()
    grid_shape = session.get_grid_dims()
    computer_player_one = BaseBot(grid_shape)
    computer_player_two = BaseBot(grid_shape)
    session.insert_bot(player=1, bot=computer_player_one)
    session.insert_bot(player=2, bot=computer_player_two)

    game = 0
    while game < 1000:
        session.run_game()
        session.reset()
        game += 1