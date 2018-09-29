function [ z, res] = admm_solve_conv2D_weighted_GradCSC(b, kernels, mask, lambda_residual, lambda_prior, max_it, tol, verbose)
    kmat = kernels;     %Kernel matrix     
    %Precompute spectra for H (assumed to be convolutional)
    psf_radius = floor( [size(kmat,1)/2, size(kmat,2)/2] );
    size_x = [size(b,1) + 2*psf_radius(1), size(b,2) + 2*psf_radius(2)];
    [dhat, dhat_flat, dhatTdhat_flat] = precompute_H_hat(kmat, size_x);
    dhatT_flat = conj(dhat_flat.');
    %Size of z is now the padded array
    size_z = [size_x(1), size_x(2), size(kmat, 3)];   
    % Objective
    objective = @(v) objectiveFunction( v, dhat, b, mask, lambda_residual, lambda_prior, psf_radius );  
    %Proximal terms
    conv_term = @(xi_hat, gammas) solve_conv_term(dhat_flat, dhatT_flat, dhatTdhat_flat, xi_hat, gammas, size_z);
    %Prox for masked data
    [M, Mtb] = precompute_MProx(b, mask, psf_radius); %M is MtM
    ProxDataMasked = @(u, theta) (Mtb + 1/theta * u ) ./ ( M + 1/theta * ones(size_x) );     
    %Prox for sparsity
    temp_num = 0.5;temp_num2 = 5;
    ProxSparse = @(u, theta) max( 0, 1 - theta./ (abs(u).*(abs(u).^temp_num+temp_num2))) .* u;   %
    %Pack lambdas and find algorithm params
    lambda = [lambda_residual, lambda_prior];
    gamma_heuristic =600* lambda_prior * 1/max(b(:));
    gamma = [gamma_heuristic/100, gamma_heuristic];
    %Initialize variables
    varsize = {size_x, size_z};
    xi = { zeros(varsize{1}), zeros(varsize{2}) };
    xi_hat = { zeros(varsize{1}), zeros(varsize{2}) };
    u = { zeros(varsize{1}), zeros(varsize{2}) };    d = { zeros(varsize{1}), zeros(varsize{2}) };    v = { zeros(varsize{1}), zeros(varsize{2}) };
    %Initial iterate
    z = zeros(varsize{2});
    z_hat = zeros(varsize{2});    
    %Display it.
    if strcmp(verbose, 'all')  
        iterate_fig = figure(222);
        Dz = real(ifft2( sum( dhat .* z_hat, 3) ));
        Dz = Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2),:);
        imagesc(Dz), axis image, colormap gray; title(sprintf('Local iterate %d',0));
    end
    %Iterate
    for i = 1:max_it
        
        %Compute v_i = H_i * z
        v{1} = real(ifft2( sum( dhat .* z_hat, 3)));
        v{2} = z;
        
        %Compute proximal updates
        u{1} = ProxDataMasked( v{1} - d{1}, lambda(1)/gamma(1) );
        u{2} = ProxSparse( v{2} - d{2}, lambda(2)/gamma(2) );
        
        for c = 1:2
            %Update running errors
            d{c} = d{c} - (v{c} - u{c});

            %Compute new xi and transform to fft
            xi{c} = u{c} + d{c};
            xi_hat{c} = fft2(xi{c});
        end

        %Solve convolutional inverse
        % z = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
        zold = z;
        z_hat = conv_term( xi_hat, gamma );
        z = real(ifft2( z_hat ));
       % figure(1002);imshow([z(:,:,3),z(:,:,13);z(:,:,23),z(:,:,33)],[])
        
        %Display it.
        if strcmp(verbose, 'all')
            
            figure(iterate_fig);
            Dz = real(ifft2( sum( dhat .* z_hat, 3) ));
            Dz = Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2),:);
            imagesc(Dz), axis image, colormap gray; title(sprintf('Local iterate %d',i));
        end

        z_diff = z - zold;
        z_comp = z;
        if norm(z_diff(:),2)/ norm(z_comp(:),2) < tol
            break;
        end
    end
    
    Dz = real(ifft2( sum( dhat .* z_hat, 3)));
    res = Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2), : );
    
return;

function [M, Mtb] = precompute_MProx(b, mask, psf_radius)
    
    M = padarray(mask, psf_radius, 0, 'both');
    Mtb = padarray(b, psf_radius, 0, 'both') .* M;
    
return;

function [dhat, dhat_flat, dhatTdhat_flat] = precompute_H_hat(kmat, size_x )
% Computes the spectra for the inversion of all H_i
%Precompute spectra for H
dhat = zeros( [size_x(1), size_x(2), size(kmat,3)] );
for i = 1:size(kmat,3)  
    dhat(:,:,i)  = psf2otf(kmat(:,:,i), size_x);
end

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, size_x(1) * size_x(2), [] );
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,2);

return;

function z_hat = solve_conv_term(dhat, dhatT, dhatTdhat, xi_hat, gammas, size_z )   
    %Rho
    rho = gammas(2)/gammas(1);
    
    %Compute b
    b = dhatT .* repmat( reshape(xi_hat{1}, size_z(1)*size_z(2), 1).', [size(dhat,2),1] ) + rho .* reshape(xi_hat{2}, size_z(1)*size_z(2), size_z(3)).';
    
    %Invert
    scInverse = repmat( ones(size(dhatTdhat.')) ./ ( rho * ones(size(dhatTdhat.')) + dhatTdhat.' ), [size(dhat,2),1] );
    x = 1/rho *b - 1/rho * scInverse .* dhatT .* repmat( sum(conj(dhatT).*b, 1), [size(dhat,2),1] );
    
    %Final transpose gives z_hat
    z_hat = reshape(x.', size_z);
return;

function f_val = objectiveFunction( z, dhat, b, mask, lambda_residual, lambda, psf_radius )
    
    %Dataterm and regularizer
    Dz = real(ifft2( sum( dhat .* fft2(z), 3)));
    f_z = lambda_residual * 1/2 * norm( reshape( mask .* Dz(1 + psf_radius(1):end - psf_radius(1),1 + psf_radius(2):end - psf_radius(2),:) - mask .* b, [], 1) , 2 )^2;
    g_z = lambda * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
return;
