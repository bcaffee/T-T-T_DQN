from tkinter import *
import tkinter as tk
from TicTacToeEnvironment import TicTacToeEnvironment
from PIL import Image, ImageTk

# DEPRECATED
class TkinterUI(tk.Frame):

    # File dialogue/browser for upload files through a button
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.createMenu()

    # Create the game of tic tac toe of the player vs. the agent (ice to have a scalable option like 4 by 4)
    def startGame(self):
        game = Tk()
        game.title("Tic Tac Toe")
        game.geometry("435x435")

        ticTacToeGame = TicTacToeEnvironment()

        self.createBoard()

        gameNotFinished = False

        # while not gameNotFinished:

    def createMenu(self):

        quitGame = Button(self)
        quitGame["text"] = "Quit"
        quitGame["command"] = self.quit
        quitGame["fg"] = "red"
        quitGame.pack({"side": "right"})

        start = Button(self)
        start["text"] = "Start",
        start["fg"] = "green"
        start["command"] = self.startGame
        start.pack({"side": "left"})

    def createBoard(self):

        # Image for tic tac toe board

        # board = PhotoImage(file="./ImagesForTK/board.png")
        # canvas.create_image(20, 20, anchor=CENTER, image=board)
        # img.render()

        # o = PhotoImage(file="./ImagesForTK/OImage.png")
        # x = PhotoImage(file="./ImagesForTK/XImage.png")

        # path = "/TicTacToeReinforcedLearning/ImagesForTK/OImage.png"
#
        # image1 = Image.open(path)
        # # photo = image1.resize(380, 450)
        # photo = ImageTk.PhotoImage(image1)
#
        # img = tk.Label(self, image=photo, cursor="dot")
        # img.image = photo
        # img.pack()
        canvas.create_line(abs(4))
        load = Image.open("")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

        # If player agent then x, if player human then 0
        topLeft = Button(self)
        topLeft["command"] = canvas.create_image(20, 20, anchor=TO, image=board)
        topLeft.pack({"side": "top"})
        topRight = Button(self)
        topRight["command"] = canvas.create_image(20, 20, anchor=NW, image=board)
        topMiddle = Button(self)
        topMiddle["command"] = canvas.create_image(20, 20, anchor=NW, image=board)

        middleLeft = Button(self)
        middleLeft["command"] = canvas.create_image(20, 20, anchor=NW, image=board)
        middleRight = Button(self)
        middleRight["command"] = canvas.create_image(20, 20, anchor=NW, image=board)
        center = Button(self)
        center["command"] = canvas.create_image(20, 20, anchor=NW, image=board)

        bottomLeft = Button(self)
        bottomLeft["command"] = canvas.create_image(20, 20, anchor=NW, image=board)
        bottomRight = Button(self)
        bottomRight["command"] = canvas.create_image(20, 20, anchor=NW, image=board)
        bottomMiddle = Button(self)
        bottomMiddle["command"] = canvas.create_image(20, 20, anchor=NW, image=board)


def main():
    menu = Tk()
    menu.title("Menu")
    menu.columnconfigure(0, weight=1)
    menu.rowconfigure(0, weight=1)
    app = TkinterUI(master=menu)
    app.mainloop()
    menu.destroy()

if __name__ == "__main__":
    main()