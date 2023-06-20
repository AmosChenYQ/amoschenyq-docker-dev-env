import numpy as np
fake_input = np.random.rand(224, 224, 3)

with open("ssd-input.npy", "wb") as f:
    np.save(f, fake_input)

with open("ssd-input.npy", "rb") as f:
    loaded_input = np.load(f)

np.testing.assert_allclose(fake_input, loaded_input)