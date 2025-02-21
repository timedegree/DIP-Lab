from skimage import io, data
import matplotlib.pyplot as plt


img_coffee = data.coffee()
img_astronaut = data.astronaut()
img_chelesa = data.chelsea()
img_sea = io.imread("./sea.jpg")


plt.figure(figsize=(8,9))
plt.subplot(3,2,1)
plt.title("origin astronaut")
plt.imshow(img_astronaut)
plt.subplot(3,2,2)
plt.title("origin sea")
plt.imshow(img_sea)
plt.subplot(3,2,3)
plt.title("origin coffee")
plt.imshow(img_coffee)
plt.subplot(3,2,4)
plt.title("Gray coffee (R channel)")
plt.imshow(img_coffee[:,:,1], cmap=plt.cm.gray)
plt.subplot(3,2,5)
plt.title("origin chelsea")
plt.imshow(img_chelesa)
plt.subplot(3,2,6)
plt.title("Gray chelsea (G channel)")
plt.imshow(img_chelesa[:,:,2], cmap=plt.cm.gray)
plt.show()

io.imsave("./coffee_gray.gif", img_coffee[:,:,1])
io.imsave("./chelsea_gray.gif", img_chelesa[:,:,2])
