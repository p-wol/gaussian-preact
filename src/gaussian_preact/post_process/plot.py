import matplotlib.pyplot as plt

def build_label(param):
    if param['act_function'] == 'weibull':
        return r'$\phi_{{\theta}}$, $\theta = {:.2f}$'.format(param['act_theta'])
    elif 'logexp' in param['act_function']:
        omega = extract_param(param['act_function'], 'omega-')
        return r'$\varphi_{{0.99, \omega}}$, $\omega = {:.2f}$'.format(omega)
    elif param['act_function'] == 'relu':
        return r'$\mathrm{{ReLU}}$'
    elif param['act_function'] == 'tanh':
        return r'$\tanh$'
    else:
        raise ValueError('Error: unrecognized activation function: {}'.format(param['act_function']))

def extract_param(s, subs):
    i0 = s.find(subs) + len(subs)
    i_max = s[i0:].rfind('_')
    if i_max == -1:
        return float(s[i0:])
    else:
        return float(s[i0:i0 + i_max])
        
def build_color(param):
    cmap = plt.get_cmap('hot')
    cmap_le = plt.get_cmap('RdPu')
    lst_theta_select = [2.05, 2.50, 3.00, 4.00, 5.00, 7.00, 10.00]
    nb_thetas = len(lst_theta_select)
    lst_omega_select = [2.00, 3.00, 6.00]
    nb_omegas = len(lst_omega_select)
    
    if param['act_function'] == 'weibull':
        i_theta = lst_theta_select.index(param['act_theta'])
        return cmap(.5 * i_theta/(nb_thetas - 1))
    elif 'logexp' in param['act_function']:
        i_omega = lst_omega_select.index(extract_param(param['act_function'], 'omega-'))
        return cmap_le(1 - .5 * i_omega/(nb_omegas - 1))
    elif param['act_function'] == 'relu':
        return 'darkturquoise'
    elif param['act_function'] == 'tanh':
        return 'fuchsia'
    else:
        raise ValueError('Error: unrecognized activation function: {}'.format(param['act_function']))
        
def build_linestyle(param):
    if param['act_function'] in ['weibull', 'relu', 'tanh']:
        return '-'
    elif 'logexp' in param['act_function']:
        return '--'
    else:
        raise ValueError('Error: unrecognized activation function: {}'.format(param['act_function']))
